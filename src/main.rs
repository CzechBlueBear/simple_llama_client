//! This is a translation of simple.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use std::ffi::CString;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::process::exit;
use std::str::FromStr;
use std::time::Duration;

#[derive(clap::Parser, Debug, Clone)]
struct Args {
    /// The path to the model
    #[command(subcommand)]
    model: Model,
    /// The prompt
    #[clap(short = 'p', long)]
    prompt: Option<String>,
    /// Read the prompt from a file
    #[clap(short = 'f', long, help = "prompt file to start generation")]
    file: Option<String>,
    /// set the length of the prompt + output in tokens
    #[arg(long, default_value_t = 32)]
    n_len: i32,
    /// override some parameters of the model
    #[arg(short = 'o', value_parser = parse_key_val)]
    key_value_overrides: Vec<(String, ParamOverrideValue)>,
    /// Disable offloading layers to the gpu
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[clap(long)]
    disable_gpu: bool,
    #[arg(short = 's', long, help = "RNG seed (default: 1234)")]
    seed: Option<u32>,
    #[arg(
        short = 't',
        long,
        help = "number of threads to use during generation (default: use all available threads)"
    )]
    threads: Option<i32>,
    #[arg(
        long,
        help = "number of threads to use during batch and prompt processing (default: use all available threads)"
    )]
    threads_batch: Option<i32>,
    #[arg(
        short = 'c',
        long,
        help = "size of the prompt context (default: loaded from themodel)"
    )]
    ctx_size: Option<NonZeroU32>,
}

/// Parse a single key-value pair
fn parse_key_val(s: &str) -> Result<(String, ParamOverrideValue)> {
    let pos = s
        .find('=')
        .ok_or_else(|| anyhow!("invalid KEY=value: no `=` found in `{}`", s))?;
    let key = s[..pos].parse()?;
    let value: String = s[pos + 1..].parse()?;
    let value = i64::from_str(&value)
        .map(ParamOverrideValue::Int)
        .or_else(|_| f64::from_str(&value).map(ParamOverrideValue::Float))
        .or_else(|_| bool::from_str(&value).map(ParamOverrideValue::Bool))
        .map_err(|_| anyhow!("must be one of i64, f64, or bool"))?;

    Ok((key, value))
}

#[derive(clap::Subcommand, Debug, Clone)]
enum Model {
    /// Use an already downloaded model
    Local {
        /// The path to the model. e.g. `/home/marcus/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa`
        path: PathBuf,
    },
    /// Download a model from huggingface (or use a cached version)
    #[clap(name = "hf-model")]
    HuggingFace {
        /// the repo containing the model. e.g. `TheBloke/Llama-2-7B-Chat-GGUF`
        repo: String,
        /// the model name. e.g. `llama-2-7b-chat.Q4_K_M.gguf`
        model: String,
    },
}

impl Model {
    /// Convert the model to a path - may download from huggingface
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
            Model::HuggingFace { model, repo } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create huggingface api")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}

/// Composes the prompt from two optional places: a file
/// and a string directly placed on the command line.
fn compose_prompt(path: Option<&str>, direct_prompt: Option<&str>) -> Result<String> {
    let mut composite_prompt = "".to_string();
    if let Some(filename) = path {
        if let Ok(contents) = std::fs::read_to_string(&filename) {
            composite_prompt = contents;
        } else {
            return Err(anyhow!("could not read {filename}"));
        }
    }
    if let Some(str) = direct_prompt {
        composite_prompt = composite_prompt + " " + &str;
    }
    if composite_prompt.len() == 0 {
        return Err(anyhow!("no prompt specified"));
    }
    Ok(composite_prompt)
}

/// Prints out the already tokenized prompt token-by-token.
fn print_tokenized_prompt(model: &LlamaModel, tokens_list: Vec<LlamaToken>) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!();
    for token in &tokens_list {
        eprint!("{}", model.token_to_str(*token, Special::Tokenize)?);
    }
    std::io::stderr().flush()?;
    Ok(())
}

/// Prints out a newly generated token.
fn print_token(model: &LlamaModel, utf8_decoder: &mut encoding_rs::Decoder, token: LlamaToken) -> Result<(), anyhow::Error> {

    // convert the result token back to human-readable bytes
    let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;

    // convert it from bytes to safe printable UTF-8
    let mut output_string = String::with_capacity(32);
    let _decode_result = utf8_decoder.decode_to_string(&output_bytes, &mut output_string, false);

    // print it out (and flush because it's usually not a complete line
    // but we want the result to be visible immediately)
    print!("{output_string}");
    std::io::stdout().flush()?;

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    let Args {
        n_len,
        model,
        prompt,
        file,
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        disable_gpu,
        key_value_overrides,
        seed,
        threads,
        threads_batch,
        ctx_size,
    } = Args::parse();

    // init LLM
    let backend = LlamaBackend::init()?;

    // offload all layers to the gpu
    let model_params = {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if !disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        LlamaModelParams::default()
    };

    // compose the prompt from (optional) file and command line
    let prompt = compose_prompt(file.as_deref(), prompt.as_deref())?;

    let mut model_params = pin!(model_params);

    for (k, v) in &key_value_overrides {
        let k = CString::new(k.as_bytes()).with_context(|| format!("invalid key: {k}"))?;
        model_params.as_mut().append_kv_override(k.as_c_str(), *v);
    }

    let model_path = model
        .get_or_load()
        .with_context(|| "failed to get model from args")?;

    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .unwrap_or_else(|_| {
            eprintln!("error: could not load model: {}", model_path.to_string_lossy());
            exit(1);
        });

    // create the context and initialize it with settings from args
    let mut ctx_params = LlamaContextParams::default();
    if ctx_size.is_some() {
        ctx_params = ctx_params.with_n_ctx(ctx_size);
    }
    if let Some(threads) = threads {
        ctx_params = ctx_params.with_n_threads(threads);
    }
    if let Some(threads_batch) = threads_batch.or(threads) {
        ctx_params = ctx_params.with_n_threads_batch(threads_batch);
    }

    let mut ctx = model.new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    // tokenize the whole prompt according to the model's rules
    let tokens_list = model.str_to_token(&prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt}"))?;

    let n_cxt = ctx.n_ctx() as i32;
    let n_kv_req = tokens_list.len() as i32 + (n_len - tokens_list.len() as i32);

    eprintln!("n_len = {n_len}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if n_kv_req > n_cxt {
        bail!(
            "n_kv_req > n_ctx, the required kv cache size is not big enough
either reduce n_len or increase n_ctx"
        )
    }

    if tokens_list.len() >= usize::try_from(n_len)? {
        bail!("the prompt is too long, it has more tokens than n_len")
    }

    // print the prompt token-by-token (mostly for debugging)
    let _ = print_tokenized_prompt(&model, tokens_list.clone());

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding
    let mut batch = LlamaBatch::new(512, 1);

    // build the first batch that contains the whole prompt;
    // llama_decode() will then process it
    let last_index: i32 = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        batch.add(
            token, i, &[0],
            i == last_index         // we want to know logits for the last token only
        )?;
    }

    // process the batch, produce the next token and logits
    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    // --- main loop ---
    let mut n_cur = batch.n_tokens();
    let mut tokens_decoded = 0;

    let t_main_start = ggml_time_us();  // for timing

    // UTF-8 decoder for safe printing of byte sequences into utf-8
    let mut utf8_decoder = encoding_rs::UTF_8.new_decoder();

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(seed.unwrap_or(1234)),
        LlamaSampler::greedy(),
    ]);

    while n_cur <= n_len {

        // use the sampler to choose the next token from the options predicted by the model
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        // stop when the model gives us EOG (End of Generation)
        if model.is_eog_token(token) {
            eprintln!();    // ensure a newline
            break;
        }

        // print it
        print_token(&model, &mut utf8_decoder, token)?;

        // prepare a new batch containing only the freshly generated token
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;

        n_cur += 1;

        ctx.decode(&mut batch).with_context(|| "failed to eval")?;

        tokens_decoded += 1;
    }

    eprintln!("\n");

    // --- finish ---
    // write out final statistics
    let t_main_end = ggml_time_us();
    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
    eprintln!(
        "decoded {} tokens in {:.2} s, speed {:.2} t/s\n",
        tokens_decoded,
        duration.as_secs_f32(),
        tokens_decoded as f32 / duration.as_secs_f32()
    );
    println!("{}", ctx.timings());

    Ok(())
}
