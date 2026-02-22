use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

const WARMUP: usize = 4;
const TIME_QUOTA: Duration = Duration::from_millis(300);
const MIN_MEASUREMENTS: usize = 3;

struct BenchResult {
    name: String,
    wall_per_run: Duration,
    runs: usize,
}

fn bench<F: FnMut()>(name: &str, mut f: F) -> BenchResult {
    // Warmup
    for _ in 0..WARMUP {
        f();
    }

    // Adaptive batching: start with batch_size=1, scale up until each batch
    // takes at least 2ms of wall time, then collect measurements for ~0.3s.
    let mut batch_size: usize = 1;
    let mut measurements: Vec<Duration> = Vec::new();
    let bench_start = Instant::now();

    loop {
        let start = Instant::now();
        for _ in 0..batch_size {
            f();
        }
        let elapsed = start.elapsed();

        if elapsed.as_secs_f64() < 0.002 {
            // Batch too fast, scale up
            batch_size = (batch_size as f64 * 1.3).ceil().max((batch_size + 1) as f64) as usize;
            continue;
        }

        let per_run = elapsed / batch_size as u32;
        measurements.push(per_run);

        let total_elapsed = bench_start.elapsed();
        if measurements.len() >= MIN_MEASUREMENTS && total_elapsed >= TIME_QUOTA {
            break;
        }

        batch_size = (batch_size as f64 * 1.3).ceil().max((batch_size + 1) as f64) as usize;
    }

    // Compute average
    let total: Duration = measurements.iter().sum();
    let avg = total / measurements.len() as u32;

    BenchResult {
        name: name.to_string(),
        wall_per_run: avg,
        runs: measurements.len(),
    }
}

fn format_duration(d: Duration) -> String {
    let nanos = d.as_nanos() as f64;
    if nanos < 1_000.0 {
        format!("{:.2}ns", nanos)
    } else if nanos < 1_000_000.0 {
        format!("{:.2}μs", nanos / 1_000.0)
    } else if nanos < 1_000_000_000.0 {
        format!("{:.2}ms", nanos / 1_000_000.0)
    } else {
        format!("{:.2}s", nanos / 1_000_000_000.0)
    }
}

fn run_suite(label: &str, tokenizer: &Tokenizer, short_text: &str, long_text: &str) {
    let batch_32: Vec<&str> = vec![short_text; 32];

    // Pre-compute decode input
    let encoding = tokenizer
        .encode(long_text, false)
        .expect("encode for decode input");
    let decode_ids: Vec<u32> = encoding.get_ids().to_vec();

    let results = vec![
        bench(&format!("{}/Encode/single_short", label), || {
            tokenizer.encode(short_text, false).unwrap();
        }),
        bench(&format!("{}/Encode/single_long", label), || {
            tokenizer.encode(long_text, false).unwrap();
        }),
        bench(&format!("{}/Encode/batch_32", label), || {
            tokenizer
                .encode_batch(batch_32.clone(), false)
                .unwrap();
        }),
        bench(&format!("{}/Decode/long", label), || {
            tokenizer.decode(decode_ids.as_slice(), true).unwrap();
        }),
    ];

    for r in &results {
        println!(
            "  {:<35} {:>10}  ({} samples)",
            r.name,
            format_duration(r.wall_per_run),
            r.runs
        );
    }
}

fn main() {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");

    let short_text =
        fs::read_to_string(data_dir.join("news_1k.txt")).expect("read news_1k.txt");
    let long_text =
        fs::read_to_string(data_dir.join("wiki_64k.txt")).expect("read wiki_64k.txt");

    println!("Rust-native HuggingFace tokenizers benchmark");
    println!("=============================================\n");

    let tokenizers = [
        ("GPT-2", "gpt2.json"),
        ("BERT-base", "bert_base.json"),
        ("LLaMA", "llama.json"),
    ];

    for (label, filename) in &tokenizers {
        let path = data_dir.join(filename);
        let tokenizer =
            Tokenizer::from_file(&path).unwrap_or_else(|e| {
                panic!("Failed to load {}: {}", path.display(), e)
            });
        println!("{}:", label);
        run_suite(label, &tokenizer, &short_text, &long_text);
        println!();
    }
}
