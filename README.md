simple_llama_client
===================

This is a simple llama.cpp client in Rust, based on an example from llama-rust-2 - currently the only difference is that it is a separate package that has llama.cpp-rs-2 as dependency but still links statically with llama.cpp so the executable is independent.

I am planning to extend it a bit for logged conversations etc. but please don't expect any breakthroughs here; the main purpose of this project is for myself to understand how these things work.
