use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel::{Receiver, Sender, bounded};
use opencv::{core, imgcodecs, imgproc, prelude::*, videoio};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

const OUTPUT: &str = "";
/// Mean squared error between frames. Lower = more sensitive. Range: 0 to ~65025.
const MSE_THRESHOLD: f64 = 220.0;
/// Structural similarity (1 = identical). Lower = more sensitive. Range: -1 to 1.
const SSIM_THRESHOLD: f64 = 0.85;
/// % of pixels differing by >30 grayscale levels. Lower = more sensitive. Range: 0 to 100.
const DIFF_THRESHOLD: f64 = 2.5;
/// % of unmatched edges (e.g. subtitles). Lower = more sensitive. Range: 0 to 100.
const EDGE_THRESHOLD: f64 = 65.0;
/// % of frame area for largest contiguous difference region. Lower = more sensitive. Range: 0 to 100.
const BLOB_THRESHOLD: f64 = 0.45;
/// Min local SSIM in any region (secondary trigger). Higher = more sensitive. Range: -1 to 1.
const LOCAL_SSIM_THRESHOLD: f64 = 0.0;
const SAVE_FRAMES: usize = 500;
const NO_IMAGES: bool = false;
const BATCH_SIZE: usize = 30;
const FRAME_GROUP_SIZE: usize = 1;
const FAST_COMPARE: bool = false;
const FRAME_LOOKAHEAD: bool = true;
const FLUSH_INTERVAL: usize = 50;

#[derive(Debug, Clone)]
struct FrameDifference {
    frame_number: i32,
    timestamp_sec: f64,
    mse: f64,
    ssim: f64,
    diff_percentage: f64,
    edge_diff_percentage: f64,
    blob_diff_pixels: f64,
    min_local_ssim: f64,
}

#[derive(Debug, Clone)]
struct VideoMeta {
    path: String,
    fps: f64,
    total_frames: i32,
    width: i32,
    height: i32,
}

#[derive(Debug, Clone, Copy)]
struct Thresholds {
    mse: f64,
    ssim: f64,
    diff_pct: f64,
    edge_diff_pct: f64,
    blob_diff_pct: f64,
    local_ssim: f64,
}

#[derive(Debug, Clone, Copy)]
struct GroupProcessingConfig {
    frame_group_size: usize,
    thresholds: Thresholds,
    frame_area: f64,
    save_diff_frames: bool,
    max_diff_frames_to_save: usize,
}

#[derive(Debug, Clone)]
struct CompareConfig {
    thresholds: Thresholds,
    save_diff_frames: bool,
    max_diff_frames_to_save: usize,
    num_workers: Option<usize>,
    batch_size: usize,
    frame_group_size: usize,
    fast_compare: bool,
    frame_lookahead: bool,
    flush_interval: usize,
}

#[derive(Debug, Clone)]
struct FrameGroupDifference {
    start_frame: i32,
    end_frame: i32,
    start_timestamp_sec: f64,
    end_timestamp_sec: f64,
    avg_mse: f64,
    avg_ssim: f64,
    avg_diff_percentage: f64,
    avg_edge_diff_percentage: f64,
    avg_blob_diff_pixels: f64,
    avg_min_local_ssim: f64,
    individual_frames: Vec<FrameDifference>,
}

#[derive(Debug)]
struct BatchFrame {
    frame_num: i32,
    frame1: Mat,
    frame2: Arc<Mat>, // video2[N]; shared so prev/next don't clone pixel data
    frame2_prev: Option<Arc<Mat>>, // video2[N-1] for lookahead
    frame2_next: Option<Arc<Mat>>, // video2[N+1] for lookahead
}

#[derive(Debug)]
struct FrameSave {
    frame_num: i32,
    frame1: Mat,
    frame2: Mat,
}

/// message sent from the frame reader thread to the processing thread
struct FrameBatch {
    frames: Vec<BatchFrame>,
    frame_data: HashMap<i32, (Mat, Mat)>,
    current_frame_num: i32,
}

#[derive(Parser, Debug)]
#[command(
    name = "kensa",
    about = "Compare two videos frame-by-frame and identify differences.",
    version = concat!(env!("CARGO_PKG_VERSION"), "-alpha+", env!("GIT_COMMIT_HASH"))
)]
struct Cli {
    video1: String,
    video2: String,
    #[arg(short, long, default_value = OUTPUT)]
    output: String,
    #[arg(long, default_value_t = MSE_THRESHOLD)]
    mse_threshold: f64,
    #[arg(long, default_value_t = SSIM_THRESHOLD)]
    ssim_threshold: f64,
    #[arg(long = "diff-threshold", default_value_t = DIFF_THRESHOLD)]
    diff_threshold: f64,
    /// edge difference threshold (percentage of unmatched edges to flag as different)
    #[arg(long = "edge-threshold", default_value_t = EDGE_THRESHOLD)]
    edge_threshold: f64,
    /// blob threshold: % of frame area for largest contiguous diff region to flag as significant
    #[arg(long = "blob-threshold", default_value_t = BLOB_THRESHOLD)]
    blob_threshold: f64,
    /// local SSIM threshold: minimum allowed local-structure similarity
    #[arg(long = "local-ssim-threshold", default_value_t = LOCAL_SSIM_THRESHOLD)]
    local_ssim_threshold: f64,
    #[arg(long = "save-frames", default_value_t = SAVE_FRAMES)]
    save_frames: usize,
    #[arg(long = "no-images", default_value_t = NO_IMAGES)]
    no_images: bool,
    #[arg(short, long)]
    workers: Option<usize>,
    #[arg(short, long, default_value_t = BATCH_SIZE)]
    batch_size: usize,
    #[arg(long, default_value_t = FRAME_GROUP_SIZE)]
    frame_group_size: usize,
    /// compare frames at half resolution for faster processing (4x fewer pixels)
    #[arg(long = "fast-compare", default_value_t = FAST_COMPARE)]
    fast_compare: bool,
    /// compare video1[N] against video2[N-1], video2[N], and video2[N+1] to handle timing offsets
    #[arg(long = "frame-lookahead", default_value_t = FRAME_LOOKAHEAD)]
    frame_lookahead: bool,
    /// flush diff images to disk after this many are queued (reduces memory usage)
    #[arg(long = "flush-interval", default_value_t = FLUSH_INTERVAL)]
    flush_interval: usize,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Cli::parse();

    if let Some(workers) = args.workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build_global()
            .context("failed to configure worker thread pool")?;
    }

    let output_dir = if args.output.trim().is_empty() {
        let v1 = Path::new(&args.video1)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("video1");
        let v2 = Path::new(&args.video2)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("video2");
        format!("./video_comparison_results_{}_{}", v1, v2)
    } else {
        args.output.clone()
    };

    let config = CompareConfig {
        thresholds: Thresholds {
            mse: args.mse_threshold,
            ssim: args.ssim_threshold,
            diff_pct: args.diff_threshold,
            edge_diff_pct: args.edge_threshold,
            blob_diff_pct: args.blob_threshold,
            local_ssim: args.local_ssim_threshold,
        },
        save_diff_frames: !args.no_images,
        max_diff_frames_to_save: args.save_frames,
        num_workers: args.workers,
        batch_size: args.batch_size,
        frame_group_size: args.frame_group_size,
        fast_compare: args.fast_compare,
        frame_lookahead: args.frame_lookahead,
        flush_interval: args.flush_interval,
    };

    let summary = compare_videos(&args.video1, &args.video2, &output_dir, &config)?;

    let results_path = Path::new(&output_dir).join("comparison_results.json");
    let json = serde_json::to_string_pretty(&summary)?;
    fs::write(&results_path, json)?;

    println!("\n{:=<20}", "=");
    println!("COMPARISON COMPLETE");
    println!("{:=<20}", "=");
    println!("Frames compared: {}", summary.comparison.frames_compared);
    println!("Frame group size: {}", summary.comparison.frame_group_size);
    println!("Total groups: {}", summary.comparison.total_groups);
    println!(
        "Significant groups: {}",
        summary.comparison.significant_groups
    );
    if summary.comparison.frames_compared > 0 {
        println!(
            "Significant frames: {}",
            summary.comparison.significant_frames
        );
        println!(
            "Percentage different: {:.2}%",
            summary.comparison.percentage_different
        );
        println!("\nAverage metrics across all frames:");
        println!(
            "  MSE:  {:.2} (lower = more similar)",
            summary.statistics.mse.mean
        );
        println!(
            "  SSIM: {:.4} (higher = more similar, max=1.0)",
            summary.statistics.ssim.mean
        );
        println!(
            "  Diff: {:.2}% of pixels differ",
            summary.statistics.diff_percentage.mean
        );
    }
    println!("\nResults saved to: {}", output_dir);

    Ok(())
}

fn compare_videos(
    video1_path: &str,
    video2_path: &str,
    output_dir: &str,
    config: &CompareConfig,
) -> Result<Summary> {
    let num_workers = config.num_workers.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    if config.frame_group_size == 0 {
        return Err(anyhow::anyhow!("frame_group_size must be >= 1"));
    }

    fs::create_dir_all(output_dir)?;
    let output_path = Path::new(output_dir);

    let mut cap1 = videoio::VideoCapture::from_file(video1_path, videoio::CAP_ANY)
        .with_context(|| format!("cannot open video 1: {video1_path}"))?;
    let mut cap2 = videoio::VideoCapture::from_file(video2_path, videoio::CAP_ANY)
        .with_context(|| format!("cannot open video 2: {video2_path}"))?;

    if !cap1.is_opened()? {
        return Err(anyhow::anyhow!("cannot open video 1: {video1_path}"));
    }
    if !cap2.is_opened()? {
        return Err(anyhow::anyhow!("cannot open video 2: {video2_path}"));
    }

    let fps1 = cap1.get(videoio::CAP_PROP_FPS)?;
    let fps2 = cap2.get(videoio::CAP_PROP_FPS)?;
    let total_frames1 = cap1.get(videoio::CAP_PROP_FRAME_COUNT)? as i32;
    let total_frames2 = cap2.get(videoio::CAP_PROP_FRAME_COUNT)? as i32;
    let width1 = cap1.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height1 = cap1.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let width2 = cap2.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height2 = cap2.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    println!("Video 1: {total_frames1} frames, {fps1:.2} fps, {width1}x{height1}");
    println!("Video 2: {total_frames2} frames, {fps2:.2} fps, {width2}x{height2}");

    let mut warnings = Vec::new();
    if (fps1 - fps2).abs() > 0.1 {
        warnings.push(format!("FPS mismatch: {fps1:.2} vs {fps2:.2}"));
    }
    if total_frames1 != total_frames2 {
        warnings.push(format!(
            "Frame count mismatch: {total_frames1} vs {total_frames2}"
        ));
    }
    if (width1, height1) != (width2, height2) {
        warnings.push(format!(
            "Resolution mismatch: {width1}x{height1} vs {width2}x{height2}"
        ));
    }

    for warning in &warnings {
        println!("WARNING: {warning}");
    }

    let min_frames = total_frames1.min(total_frames2);
    println!(
        "\nComparing {min_frames} frames using {num_workers} workers (batch size: {})...",
        config.batch_size
    );
    if config.frame_group_size > 1 {
        println!(
            "Using frame group averaging with group size: {}",
            config.frame_group_size
        );
    }
    if config.fast_compare {
        println!("Fast compare mode: comparing at half resolution (4x faster)");
    }
    if config.frame_lookahead {
        println!(
            "Frame lookahead mode: comparing against adjacent frames to handle timing offsets"
        );
    }

    let mut all_differences: Vec<FrameDifference> = Vec::new();
    let mut significant_groups: Vec<FrameGroupDifference> = Vec::new();
    let mut frames_to_save: Vec<FrameSave> = Vec::new();
    let mut pending_diffs: Vec<FrameDifference> = Vec::new();
    let mut pending_frame_data: HashMap<i32, (Mat, Mat)> = HashMap::new();

    let pixel_threshold = 30i32;

    let mut frame_area = (width1 as f64) * (height1 as f64);
    if config.fast_compare {
        frame_area /= 4.0;
    }

    let group_config = GroupProcessingConfig {
        frame_group_size: config.frame_group_size,
        thresholds: config.thresholds,
        frame_area,
        save_diff_frames: config.save_diff_frames,
        max_diff_frames_to_save: config.max_diff_frames_to_save,
    };

    // create bounded channel for producer-consumer pattern (2 slots for double buffering)
    let (tx, rx): (Sender<FrameBatch>, Receiver<FrameBatch>) = bounded(2);

    // shared counter for frames saved - reader thread stops cloning when max is reached
    let frames_saved_count = Arc::new(AtomicUsize::new(0));
    let frames_saved_count_reader = Arc::clone(&frames_saved_count);
    let max_diff_frames = config.max_diff_frames_to_save;

    // spawn dedicated reader thread to decouple video decoding from processing
    let batch_size = config.batch_size;
    let save_diff_frames = config.save_diff_frames;
    let frame_lookahead = config.frame_lookahead;
    let reader_handle = thread::spawn(move || -> Result<()> {
        let mut frame_num = 0i32;
        // for frame_lookahead: keep the last video2 frame from previous batch (Arc = no pixel copy)
        let mut prev_batch_last_frame2: Option<Arc<Mat>> = None;

        while frame_num < min_frames {
            let mut batch_frame_data: HashMap<i32, (Mat, Mat)> = HashMap::new();

            // read all frame pairs; store each video2 frame once in Arc for shared prev/curr/next
            let mut raw_pairs: Vec<(i32, Mat, Arc<Mat>)> = Vec::new();
            let mut frame2_frames: Vec<Arc<Mat>> = Vec::new();

            for _ in 0..batch_size {
                let mut raw1 = Mat::default();
                let mut raw2 = Mat::default();
                let ret1 = cap1.read(&mut raw1)?;
                let ret2 = cap2.read(&mut raw2)?;
                if !ret1 || !ret2 {
                    break;
                }

                let frame1 = raw1;
                let mut frame2 = raw2;
                if frame1.size()? != frame2.size()? {
                    let mut resized = Mat::default();
                    imgproc::resize(
                        &frame2,
                        &mut resized,
                        core::Size::new(frame1.cols(), frame1.rows()),
                        0.0,
                        0.0,
                        imgproc::INTER_LINEAR,
                    )?;
                    frame2 = resized;
                }

                // clone frames for saving if enabled AND we haven't hit the max yet
                if save_diff_frames
                    && frames_saved_count_reader.load(Ordering::Relaxed) < max_diff_frames
                {
                    let frame1_clone = frame1.try_clone()?;
                    let frame2_clone = frame2.try_clone()?;
                    batch_frame_data.insert(frame_num, (frame1_clone, frame2_clone));
                }

                let frame2_arc = Arc::new(frame2);
                frame2_frames.push(frame2_arc.clone());
                raw_pairs.push((frame_num, frame1, frame2_arc));
                frame_num += 1;
            }

            if raw_pairs.is_empty() {
                break;
            }

            // build BatchFrames: every frame gets prev/next when available (Arc = no extra pixel memory)
            let mut batch_data: Vec<BatchFrame> = Vec::new();

            for (i, (fnum, frame1, frame2)) in raw_pairs.into_iter().enumerate() {
                let frame2_prev = if frame_lookahead {
                    if i == 0 {
                        prev_batch_last_frame2.clone()
                    } else {
                        Some(frame2_frames[i - 1].clone())
                    }
                } else {
                    None
                };
                let frame2_next = if frame_lookahead {
                    frame2_frames.get(i + 1).cloned()
                } else {
                    None
                };

                batch_data.push(BatchFrame {
                    frame_num: fnum,
                    frame1,
                    frame2,
                    frame2_prev,
                    frame2_next,
                });
            }

            if frame_lookahead {
                prev_batch_last_frame2 = frame2_frames.last().cloned();
            }

            let batch = FrameBatch {
                frames: batch_data,
                frame_data: batch_frame_data,
                current_frame_num: frame_num,
            };

            // send batch to processing thread (blocks if channel is full - backpressure)
            if tx.send(batch).is_err() {
                break; // receiver dropped, stop reading
            }
        }

        cap1.release()?;
        cap2.release()?;
        Ok(())
    });

    // process batches as they arrive from the reader thread
    let mse_threshold = config.thresholds.mse;
    let fast_compare = config.fast_compare;
    let frame_lookahead = config.frame_lookahead;
    for batch in rx {
        let batch_results: Vec<FrameDifference> = batch
            .frames
            .par_iter()
            .map(|item| {
                process_frame_pair(
                    item,
                    fps1,
                    pixel_threshold,
                    mse_threshold,
                    fast_compare,
                    frame_lookahead,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        all_differences.extend(batch_results.iter().cloned());

        let mut combined_diffs = Vec::new();
        combined_diffs.extend(pending_diffs);
        combined_diffs.extend(batch_results);

        let mut combined_frame_data = pending_frame_data;
        combined_frame_data.extend(batch.frame_data);

        let (remaining_diffs, remaining_frame_data) = process_complete_groups(
            &combined_diffs,
            combined_frame_data,
            group_config,
            &mut significant_groups,
            &mut frames_to_save,
            false,
        )?;

        pending_diffs = remaining_diffs;
        pending_frame_data = remaining_frame_data;

        // periodic flush: write diff images to disk when buffer reaches threshold
        if config.save_diff_frames && frames_to_save.len() >= config.flush_interval {
            let flush_count = frames_to_save.len();
            frames_to_save
                .par_iter()
                .try_for_each(|item| save_diff_image(item, output_path))?;
            frames_to_save.clear();
            frames_saved_count.fetch_add(flush_count, Ordering::Relaxed);
            println!(
                "    Flushed {} diff images to disk (total saved: {})",
                flush_count,
                frames_saved_count.load(Ordering::Relaxed)
            );
        }

        println!(
            "  Processed {}/{min_frames} frames...",
            batch.current_frame_num
        );
    }

    // wait for reader thread to finish and check for errors
    reader_handle
        .join()
        .map_err(|_| anyhow::anyhow!("reader thread panicked"))??;

    if !pending_diffs.is_empty() {
        let _ = process_complete_groups(
            &pending_diffs,
            pending_frame_data,
            group_config,
            &mut significant_groups,
            &mut frames_to_save,
            true,
        )?;
    }

    println!(
        "\nGrouping complete. Found {} significant groups.",
        significant_groups.len()
    );

    // final flush: save any remaining diff images
    if config.save_diff_frames && !frames_to_save.is_empty() {
        let final_count = frames_to_save.len();
        frames_to_save
            .par_iter()
            .try_for_each(|item| save_diff_image(item, output_path))?;
        frames_saved_count.fetch_add(final_count, Ordering::Relaxed);
    }

    let total_saved = frames_saved_count.load(Ordering::Relaxed);
    if config.save_diff_frames && total_saved > 0 {
        println!("\nSaved {} diff visualization images total.", total_saved);
    }

    let video1_meta = VideoMeta {
        path: video1_path.to_string(),
        fps: fps1,
        total_frames: total_frames1,
        width: width1,
        height: height1,
    };
    let video2_meta = VideoMeta {
        path: video2_path.to_string(),
        fps: fps2,
        total_frames: total_frames2,
        width: width2,
        height: height2,
    };
    let summary = build_summary(
        &video1_meta,
        &video2_meta,
        config.frame_group_size,
        config.thresholds,
        frame_area,
        warnings,
        &all_differences,
        &significant_groups,
    );

    Ok(summary)
}

fn process_frame_pair(
    item: &BatchFrame,
    fps: f64,
    pixel_threshold: i32,
    mse_threshold: f64,
    fast_compare: bool,
    frame_lookahead: bool,
) -> Result<FrameDifference> {
    // compare against current frame2
    let (mse, ssim, diff_pct, edge_diff_pct, blob_diff, min_local_ssim) = compare_frames(
        &item.frame1,
        item.frame2.as_ref(),
        pixel_threshold,
        mse_threshold,
        fast_compare,
    )?;

    // lazy lookahead: only do extra comparisons if initial result shows potential issues
    // this saves compute time and the frame clones are only used when needed
    let needs_lookahead =
        frame_lookahead && (mse > mse_threshold * 0.5 || ssim < 0.98 || min_local_ssim < 0.5);

    let (
        best_mse,
        best_ssim,
        best_diff_pct,
        best_edge_diff_pct,
        best_blob_diff,
        best_min_local_ssim,
    ) = if needs_lookahead {
        // collect all comparison results: (mse, ssim, diff_pct, edge_diff_pct, blob_diff, min_local_ssim)
        let mut candidates: Vec<(f64, f64, f64, f64, f64, f64)> = vec![(
            mse,
            ssim,
            diff_pct,
            edge_diff_pct,
            blob_diff,
            min_local_ssim,
        )];

        // compare against previous frame if available
        if let Some(ref frame2_prev) = item.frame2_prev
            && let Ok((
                prev_mse,
                prev_ssim,
                prev_diff,
                prev_edge_diff,
                prev_blob,
                prev_min_local_ssim,
            )) = compare_frames(
                &item.frame1,
                frame2_prev.as_ref(),
                pixel_threshold,
                mse_threshold,
                fast_compare,
            )
        {
            candidates.push((
                prev_mse,
                prev_ssim,
                prev_diff,
                prev_edge_diff,
                prev_blob,
                prev_min_local_ssim,
            ));
        }

        // compare against next frame if available
        if let Some(ref frame2_next) = item.frame2_next
            && let Ok((
                next_mse,
                next_ssim,
                next_diff,
                next_edge_diff,
                next_blob,
                next_min_local_ssim,
            )) = compare_frames(
                &item.frame1,
                frame2_next.as_ref(),
                pixel_threshold,
                mse_threshold,
                fast_compare,
            )
        {
            candidates.push((
                next_mse,
                next_ssim,
                next_diff,
                next_edge_diff,
                next_blob,
                next_min_local_ssim,
            ));
        }

        // pick the best match (lowest MSE = most similar)
        candidates
            .into_iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((
                mse,
                ssim,
                diff_pct,
                edge_diff_pct,
                blob_diff,
                min_local_ssim,
            ))
    } else {
        (
            mse,
            ssim,
            diff_pct,
            edge_diff_pct,
            blob_diff,
            min_local_ssim,
        )
    };

    let timestamp = item.frame_num as f64 / fps;
    Ok(FrameDifference {
        frame_number: item.frame_num,
        timestamp_sec: timestamp,
        mse: best_mse,
        ssim: best_ssim,
        diff_percentage: best_diff_pct,
        edge_diff_percentage: best_edge_diff_pct,
        blob_diff_pixels: best_blob_diff,
        min_local_ssim: best_min_local_ssim,
    })
}

fn compare_frames(
    frame1: &Mat,
    frame2: &Mat,
    pixel_threshold: i32,
    mse_threshold: f64,
    fast_compare: bool,
) -> Result<(f64, f64, f64, f64, f64, f64)> {
    let mut gray1 = Mat::default();
    let mut gray2 = Mat::default();
    imgproc::cvt_color(
        frame1,
        &mut gray1,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    imgproc::cvt_color(
        frame2,
        &mut gray2,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // fast compare: downsample by 2x (4x fewer pixels to process)
    let (cmp_gray1, cmp_gray2) = if fast_compare {
        let mut small1 = Mat::default();
        let mut small2 = Mat::default();
        let new_size = core::Size::new(gray1.cols() / 2, gray1.rows() / 2);
        imgproc::resize(&gray1, &mut small1, new_size, 0.0, 0.0, imgproc::INTER_AREA)?;
        imgproc::resize(&gray2, &mut small2, new_size, 0.0, 0.0, imgproc::INTER_AREA)?;
        (small1, small2)
    } else {
        (gray1, gray2)
    };

    let mut diff = Mat::default();
    core::absdiff(&cmp_gray1, &cmp_gray2, &mut diff)?;

    let mut diff_f64 = Mat::default();
    diff.convert_to(&mut diff_f64, core::CV_64F, 1.0, 0.0)?;
    let mut diff_sq = Mat::default();
    core::pow(&diff_f64, 2.0, &mut diff_sq)?;
    let mse = core::mean(&diff_sq, &core::no_array())?[0];

    // compute diff_percentage first (needed for early-exit return)
    let mut mask = Mat::default();
    core::compare(
        &diff,
        &core::Scalar::all(pixel_threshold as f64),
        &mut mask,
        core::CmpTypes::CMP_GT as i32,
    )?;
    let diff_pixels = core::count_non_zero(&mask)? as f64;
    let total_pixels = diff.total() as f64;
    let diff_percentage = if total_pixels > 0.0 {
        (diff_pixels / total_pixels) * 100.0
    } else {
        0.0
    };

    // compute edge difference (percentage of unmatched edges)
    let edge_diff_pct = compute_edge_difference(&cmp_gray1, &cmp_gray2)?;

    // compute blob difference (largest contiguous region of difference in pixels)
    let blob_diff_pixels = compute_blob_difference(&cmp_gray1, &cmp_gray2)?;

    // early-exit: skip expensive SSIM if MSE is far below threshold (frames nearly identical)
    if mse < mse_threshold * 0.05 {
        return Ok((
            mse,
            1.0,
            diff_percentage,
            edge_diff_pct,
            blob_diff_pixels,
            1.0,
        ));
    }

    let (ssim, min_local_ssim) = compute_ssim_metrics(&cmp_gray1, &cmp_gray2)?;

    Ok((
        mse,
        ssim,
        diff_percentage,
        edge_diff_pct,
        blob_diff_pixels,
        min_local_ssim,
    ))
}

fn compute_ssim_metrics(img1: &Mat, img2: &Mat) -> Result<(f64, f64)> {
    // sSIM constants (using f64 for Scalar compatibility, but computations are f32)
    let c1 = (0.01_f64 * 255.0).powi(2);
    let c2 = (0.03_f64 * 255.0).powi(2);

    // downsample inputs to 1/4 resolution for massive memory savings
    // (1920x1080 -> 480x270 = 16x fewer pixels per Mat)
    let scale = 4;
    let small_size = core::Size::new(img1.cols() / scale, img1.rows() / scale);
    let mut img1_small = Mat::default();
    let mut img2_small = Mat::default();
    imgproc::resize(
        img1,
        &mut img1_small,
        small_size,
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    imgproc::resize(
        img2,
        &mut img2_small,
        small_size,
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;

    // use 7x7 window (standard in fast SSIM implementations)
    let window_size = core::Size::new(7, 7);
    let anchor = core::Point::new(-1, -1);

    // convert to CV_32F (single precision) instead of CV_64F - halves memory
    let mut img1_f = Mat::default();
    let mut img2_f = Mat::default();
    img1_small.convert_to(&mut img1_f, core::CV_32F, 1.0, 0.0)?;
    img2_small.convert_to(&mut img2_f, core::CV_32F, 1.0, 0.0)?;

    // drop the u8 versions immediately
    drop(img1_small);
    drop(img2_small);

    // use box filter instead of Gaussian blur (faster, similar results for SSIM)
    let mut mu1 = Mat::default();
    let mut mu2 = Mat::default();
    imgproc::box_filter(
        &img1_f,
        &mut mu1,
        -1,
        window_size,
        anchor,
        true,
        core::BORDER_DEFAULT,
    )?;
    imgproc::box_filter(
        &img2_f,
        &mut mu2,
        -1,
        window_size,
        anchor,
        true,
        core::BORDER_DEFAULT,
    )?;

    let mut mu1_sq = Mat::default();
    let mut mu2_sq = Mat::default();
    let mut mu1_mu2 = Mat::default();
    core::multiply(&mu1, &mu1, &mut mu1_sq, 1.0, -1)?;
    core::multiply(&mu2, &mu2, &mut mu2_sq, 1.0, -1)?;
    core::multiply(&mu1, &mu2, &mut mu1_mu2, 1.0, -1)?;

    let mut img1_sq = Mat::default();
    let mut img2_sq = Mat::default();
    let mut img1_img2 = Mat::default();
    core::multiply(&img1_f, &img1_f, &mut img1_sq, 1.0, -1)?;
    core::multiply(&img2_f, &img2_f, &mut img2_sq, 1.0, -1)?;
    core::multiply(&img1_f, &img2_f, &mut img1_img2, 1.0, -1)?;

    // done with img1_f, img2_f - drop to free memory
    drop(img1_f);
    drop(img2_f);

    let mut sigma1_sq = Mat::default();
    let mut sigma2_sq = Mat::default();
    let mut sigma12 = Mat::default();
    imgproc::box_filter(
        &img1_sq,
        &mut sigma1_sq,
        -1,
        window_size,
        anchor,
        true,
        core::BORDER_DEFAULT,
    )?;
    imgproc::box_filter(
        &img2_sq,
        &mut sigma2_sq,
        -1,
        window_size,
        anchor,
        true,
        core::BORDER_DEFAULT,
    )?;
    imgproc::box_filter(
        &img1_img2,
        &mut sigma12,
        -1,
        window_size,
        anchor,
        true,
        core::BORDER_DEFAULT,
    )?;

    // done with squared images
    drop(img1_sq);
    drop(img2_sq);
    drop(img1_img2);

    // sigma^2 = E[X^2] - E[X]^2
    let mut sigma1_sq_tmp = Mat::default();
    let mut sigma2_sq_tmp = Mat::default();
    let mut sigma12_tmp = Mat::default();
    core::subtract(
        &sigma1_sq,
        &mu1_sq,
        &mut sigma1_sq_tmp,
        &core::no_array(),
        -1,
    )?;
    core::subtract(
        &sigma2_sq,
        &mu2_sq,
        &mut sigma2_sq_tmp,
        &core::no_array(),
        -1,
    )?;
    core::subtract(&sigma12, &mu1_mu2, &mut sigma12_tmp, &core::no_array(), -1)?;
    sigma1_sq = sigma1_sq_tmp;
    sigma2_sq = sigma2_sq_tmp;
    sigma12 = sigma12_tmp;

    // sSIM = (2*mu1*mu2 + c1)(2*sigma12 + c2) / ((mu1^2 + mu2^2 + c1)(sigma1^2 + sigma2^2 + c2))
    let mut t1 = Mat::default();
    let mut t2 = Mat::default();
    core::add_weighted(&mu1_mu2, 2.0, &mu1_mu2, 0.0, c1, &mut t1, -1)?;
    core::add_weighted(&sigma12, 2.0, &sigma12, 0.0, c2, &mut t2, -1)?;

    drop(mu1_mu2);
    drop(sigma12);

    let mut numerator = Mat::default();
    core::multiply(&t1, &t2, &mut numerator, 1.0, -1)?;
    drop(t1);
    drop(t2);

    let mut t3 = Mat::default();
    let mut t4 = Mat::default();
    core::add(&mu1_sq, &mu2_sq, &mut t3, &core::no_array(), -1)?;
    drop(mu1_sq);
    drop(mu2_sq);
    let mut t3_tmp = Mat::default();
    core::add(
        &t3,
        &core::Scalar::all(c1),
        &mut t3_tmp,
        &core::no_array(),
        -1,
    )?;
    t3 = t3_tmp;

    core::add(&sigma1_sq, &sigma2_sq, &mut t4, &core::no_array(), -1)?;
    drop(sigma1_sq);
    drop(sigma2_sq);
    let mut t4_tmp = Mat::default();
    core::add(
        &t4,
        &core::Scalar::all(c2),
        &mut t4_tmp,
        &core::no_array(),
        -1,
    )?;
    t4 = t4_tmp;

    let mut denominator = Mat::default();
    core::multiply(&t3, &t4, &mut denominator, 1.0, -1)?;
    drop(t3);
    drop(t4);

    let mut ssim_map = Mat::default();
    core::divide2(&numerator, &denominator, &mut ssim_map, 1.0, -1)?;
    drop(numerator);
    drop(denominator);

    let ssim = core::mean(&ssim_map, &core::no_array())?[0];

    // skip local SSIM computation if global SSIM is high (frames clearly similar)
    if ssim > 0.96 {
        return Ok((ssim, 1.0));
    }

    // for local SSIM, smooth with small kernel then find minimum
    // (ssim_map is already at 1/4 resolution, no need to downsample further)
    let mut local_ssim_map = Mat::default();
    imgproc::box_filter(
        &ssim_map,
        &mut local_ssim_map,
        -1,
        core::Size::new(5, 5),
        core::Point::new(-1, -1),
        true,
        core::BORDER_DEFAULT,
    )?;
    drop(ssim_map);

    let mut min_val = 0.0;
    let mut max_val = 0.0;
    core::min_max_loc(
        &local_ssim_map,
        Some(&mut min_val),
        Some(&mut max_val),
        None,
        None,
        &core::no_array(),
    )?;

    Ok((ssim, min_val))
}

/// compute edge-based difference to detect structural elements (like subtitles)
/// that exist in one frame but not the other.
/// returns the percentage of edges that are unmatched between frames.
fn compute_edge_difference(gray1: &Mat, gray2: &Mat) -> Result<f64> {
    let mut edges1 = Mat::default();
    let mut edges2 = Mat::default();

    imgproc::canny(gray1, &mut edges1, 50.0, 150.0, 3, false)?;
    imgproc::canny(gray2, &mut edges2, 50.0, 150.0, 3, false)?;

    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(3, 3),
        core::Point::new(-1, -1),
    )?;

    let mut edges1_dilated = Mat::default();
    let mut edges2_dilated = Mat::default();
    imgproc::dilate(
        &edges1,
        &mut edges1_dilated,
        &kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_DEFAULT,
        core::Scalar::default(),
    )?;
    imgproc::dilate(
        &edges2,
        &mut edges2_dilated,
        &kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_DEFAULT,
        core::Scalar::default(),
    )?;

    let mut edge_diff = Mat::default();
    core::bitwise_xor(
        &edges1_dilated,
        &edges2_dilated,
        &mut edge_diff,
        &core::no_array(),
    )?;

    let diff_edge_pixels = core::count_non_zero(&edge_diff)? as f64;

    let mut edge_union = Mat::default();
    core::bitwise_or(
        &edges1_dilated,
        &edges2_dilated,
        &mut edge_union,
        &core::no_array(),
    )?;
    let total_edge_pixels = core::count_non_zero(&edge_union)? as f64;

    if total_edge_pixels < 100.0 {
        return Ok(0.0);
    }

    let edge_diff_pct = (diff_edge_pixels / total_edge_pixels) * 100.0;
    Ok(edge_diff_pct)
}

/// detect coherent regions (blobs) of difference between frames.
/// this catches text/graphics that appear in one frame but not the other,
/// regardless of where they appear on screen.
/// returns the size of the largest contiguous difference blob in pixels.
fn compute_blob_difference(gray1: &Mat, gray2: &Mat) -> Result<f64> {
    // compute absolute difference between frames
    let mut diff = Mat::default();
    core::absdiff(gray1, gray2, &mut diff)?;

    // threshold to create binary mask of "significantly different" pixels.
    // 35 reduces encode/compression noise while still catching missing styled text.
    let mut binary_mask = Mat::default();
    imgproc::threshold(&diff, &mut binary_mask, 35.0, 255.0, imgproc::THRESH_BINARY)?;

    // morphological operations to connect nearby pixels (text characters)
    // and reduce noise from compression artifacts
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(3, 3),
        core::Point::new(-1, -1),
    )?;

    // close operation: dilate then erode - connects nearby regions
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &binary_mask,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_DEFAULT,
        core::Scalar::default(),
    )?;

    // open operation: erode then dilate - removes small noise
    let mut cleaned = Mat::default();
    imgproc::morphology_ex(
        &closed,
        &mut cleaned,
        imgproc::MORPH_OPEN,
        &kernel,
        core::Point::new(-1, -1),
        1,
        core::BORDER_DEFAULT,
        core::Scalar::default(),
    )?;

    // find connected components (blobs) in the difference mask
    let mut labels = Mat::default();
    let mut stats = Mat::default();
    let mut centroids = Mat::default();
    let num_labels = imgproc::connected_components_with_stats(
        &cleaned,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        core::CV_32S,
    )?;

    // find the largest blob (excluding background which is label 0)
    let mut max_blob_area: f64 = 0.0;
    for label in 1..num_labels {
        let area = *stats.at_2d::<i32>(label, imgproc::CC_STAT_AREA)? as f64;
        if area > max_blob_area {
            max_blob_area = area;
        }
    }

    Ok(max_blob_area)
}

type X = (Vec<FrameDifference>, HashMap<i32, (Mat, Mat)>);
fn process_complete_groups(
    diffs: &[FrameDifference],
    mut frame_data: HashMap<i32, (Mat, Mat)>,
    config: GroupProcessingConfig,
    significant_groups: &mut Vec<FrameGroupDifference>,
    frames_to_save: &mut Vec<FrameSave>,
    is_final: bool,
) -> Result<X> {
    if diffs.is_empty() {
        return Ok((Vec::new(), HashMap::new()));
    }

    let blob_threshold_pixels = (config.thresholds.blob_diff_pct / 100.0) * config.frame_area;

    let mut idx = 0usize;
    while idx + config.frame_group_size <= diffs.len() {
        let group_frames = &diffs[idx..idx + config.frame_group_size];
        let (avg_mse, avg_ssim, avg_diff_pct, avg_edge_diff_pct, avg_blob_diff, avg_min_local_ssim) =
            average_group(group_frames);

        // primary triggers: structural issues that indicate real differences
        // local_ssim < -0.1 is a primary trigger because very negative values
        // indicate a clear localized structural difference (e.g., subtitle issues)
        let primary_trigger = avg_ssim < config.thresholds.ssim
            || avg_diff_pct > config.thresholds.diff_pct
            || avg_blob_diff > blob_threshold_pixels
            || avg_min_local_ssim < -0.1;

        // secondary triggers: metrics that can be affected by encoder noise
        // only trigger if SSIM also indicates some structural difference (< 0.97)
        let secondary_trigger = avg_ssim < 0.97
            && (avg_mse > config.thresholds.mse
                || avg_edge_diff_pct > config.thresholds.edge_diff_pct
                || avg_min_local_ssim < config.thresholds.local_ssim);

        let is_significant = primary_trigger || secondary_trigger;

        if is_significant {
            let group = FrameGroupDifference {
                start_frame: group_frames.first().unwrap().frame_number,
                end_frame: group_frames.last().unwrap().frame_number,
                start_timestamp_sec: group_frames.first().unwrap().timestamp_sec,
                end_timestamp_sec: group_frames.last().unwrap().timestamp_sec,
                avg_mse,
                avg_ssim,
                avg_diff_percentage: avg_diff_pct,
                avg_edge_diff_percentage: avg_edge_diff_pct,
                avg_blob_diff_pixels: avg_blob_diff,
                avg_min_local_ssim,
                individual_frames: group_frames.to_vec(),
            };
            significant_groups.push(group);

            if config.save_diff_frames && frames_to_save.len() < config.max_diff_frames_to_save {
                let middle_idx = group_frames[group_frames.len() / 2].frame_number;
                if let Some((f1, f2)) = frame_data.remove(&middle_idx) {
                    frames_to_save.push(FrameSave {
                        frame_num: middle_idx,
                        frame1: f1,
                        frame2: f2,
                    });
                }
            }
        }

        idx += config.frame_group_size;
    }

    let mut remaining_diffs: Vec<FrameDifference> = diffs[idx..].to_vec();

    if is_final && !remaining_diffs.is_empty() {
        let (avg_mse, avg_ssim, avg_diff_pct, avg_edge_diff_pct, avg_blob_diff, avg_min_local_ssim) =
            average_group(&remaining_diffs);

        // primary triggers: structural issues that indicate real differences
        // local_ssim < -0.1 is a primary trigger because very negative values
        // indicate a clear localized structural difference (e.g., subtitle issues)
        let primary_trigger = avg_ssim < config.thresholds.ssim
            || avg_diff_pct > config.thresholds.diff_pct
            || avg_blob_diff > blob_threshold_pixels
            || avg_min_local_ssim < -0.1;

        // secondary triggers: metrics that can be affected by encoder noise
        // only trigger if SSIM also indicates some structural difference (< 0.97)
        let secondary_trigger = avg_ssim < 0.97
            && (avg_mse > config.thresholds.mse
                || avg_edge_diff_pct > config.thresholds.edge_diff_pct
                || avg_min_local_ssim < config.thresholds.local_ssim);

        let is_significant = primary_trigger || secondary_trigger;

        if is_significant {
            let group = FrameGroupDifference {
                start_frame: remaining_diffs.first().unwrap().frame_number,
                end_frame: remaining_diffs.last().unwrap().frame_number,
                start_timestamp_sec: remaining_diffs.first().unwrap().timestamp_sec,
                end_timestamp_sec: remaining_diffs.last().unwrap().timestamp_sec,
                avg_mse,
                avg_ssim,
                avg_diff_percentage: avg_diff_pct,
                avg_edge_diff_percentage: avg_edge_diff_pct,
                avg_blob_diff_pixels: avg_blob_diff,
                avg_min_local_ssim,
                individual_frames: remaining_diffs.clone(),
            };
            significant_groups.push(group);

            if config.save_diff_frames && frames_to_save.len() < config.max_diff_frames_to_save {
                let middle_idx = remaining_diffs[remaining_diffs.len() / 2].frame_number;
                if let Some((f1, f2)) = frame_data.remove(&middle_idx) {
                    frames_to_save.push(FrameSave {
                        frame_num: middle_idx,
                        frame1: f1,
                        frame2: f2,
                    });
                }
            }
        }

        remaining_diffs.clear();
    }

    let mut remaining_frame_data = HashMap::new();
    for diff in &remaining_diffs {
        if let Some(pair) = frame_data.remove(&diff.frame_number) {
            remaining_frame_data.insert(diff.frame_number, pair);
        }
    }

    Ok((remaining_diffs, remaining_frame_data))
}

fn average_group(group_frames: &[FrameDifference]) -> (f64, f64, f64, f64, f64, f64) {
    let len = group_frames.len() as f64;
    let avg_mse = group_frames.iter().map(|f| f.mse).sum::<f64>() / len;
    let avg_ssim = group_frames.iter().map(|f| f.ssim).sum::<f64>() / len;
    let avg_diff_pct = group_frames.iter().map(|f| f.diff_percentage).sum::<f64>() / len;
    let avg_edge_diff_pct = group_frames
        .iter()
        .map(|f| f.edge_diff_percentage)
        .sum::<f64>()
        / len;
    let avg_blob_diff = group_frames.iter().map(|f| f.blob_diff_pixels).sum::<f64>() / len;
    let avg_min_local_ssim = group_frames.iter().map(|f| f.min_local_ssim).sum::<f64>() / len;
    (
        avg_mse,
        avg_ssim,
        avg_diff_pct,
        avg_edge_diff_pct,
        avg_blob_diff,
        avg_min_local_ssim,
    )
}

fn save_diff_image(item: &FrameSave, output_path: &Path) -> Result<()> {
    let vis = create_diff_visualization(&item.frame1, &item.frame2)?;
    let vis_path = output_path.join(format!("diff_frame_{:06}.png", item.frame_num));
    imgcodecs::imwrite(
        vis_path.to_string_lossy().as_ref(),
        &vis,
        &core::Vector::<i32>::new(),
    )?;
    Ok(())
}

fn create_diff_visualization(frame1: &Mat, frame2: &Mat) -> Result<Mat> {
    let mut diff = Mat::default();
    core::absdiff(frame1, frame2, &mut diff)?;

    let mut diff_amplified = Mat::default();
    core::convert_scale_abs(&diff, &mut diff_amplified, 3.0, 0.0)?;

    let mut gray_diff = Mat::default();
    imgproc::cvt_color(
        &diff,
        &mut gray_diff,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut gray_scaled = Mat::default();
    core::convert_scale_abs(&gray_diff, &mut gray_scaled, 3.0, 0.0)?;

    let mut heatmap = Mat::default();
    imgproc::apply_color_map(&gray_scaled, &mut heatmap, imgproc::COLORMAP_JET)?;

    let mut frame1_vis = frame1.try_clone()?;
    let mut frame2_vis = frame2.try_clone()?;
    let mut diff_vis = diff_amplified;
    let mut heatmap_vis = heatmap;

    let max_width = 640;
    if frame1_vis.cols() > max_width {
        let scale = max_width as f64 / frame1_vis.cols() as f64;
        let new_size = core::Size::new(max_width, (frame1_vis.rows() as f64 * scale) as i32);
        let mut resized = Mat::default();
        imgproc::resize(
            &frame1_vis,
            &mut resized,
            new_size,
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        frame1_vis = resized;

        let mut resized2 = Mat::default();
        imgproc::resize(
            &frame2_vis,
            &mut resized2,
            new_size,
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        frame2_vis = resized2;

        let mut resized3 = Mat::default();
        imgproc::resize(
            &diff_vis,
            &mut resized3,
            new_size,
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        diff_vis = resized3;

        let mut resized4 = Mat::default();
        imgproc::resize(
            &heatmap_vis,
            &mut resized4,
            new_size,
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        heatmap_vis = resized4;
    }

    let label_offset = frame1_vis.cols() + 10;

    let mut top_row = Mat::default();
    let mut bottom_row = Mat::default();
    let mut top_vec = core::Vector::<Mat>::new();
    top_vec.push(frame1_vis);
    top_vec.push(frame2_vis);
    core::hconcat(&top_vec, &mut top_row)?;

    let mut bottom_vec = core::Vector::<Mat>::new();
    bottom_vec.push(diff_vis);
    bottom_vec.push(heatmap_vis);
    core::hconcat(&bottom_vec, &mut bottom_row)?;

    let font = imgproc::FONT_HERSHEY_SIMPLEX;
    imgproc::put_text(
        &mut top_row,
        "Video 1",
        core::Point::new(10, 30),
        font,
        1.0,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;
    imgproc::put_text(
        &mut top_row,
        "Video 2",
        core::Point::new(label_offset, 30),
        font,
        1.0,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;
    imgproc::put_text(
        &mut bottom_row,
        "Difference",
        core::Point::new(10, 30),
        font,
        1.0,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;
    imgproc::put_text(
        &mut bottom_row,
        "Heatmap",
        core::Point::new(label_offset, 30),
        font,
        1.0,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;

    let mut final_vis = Mat::default();
    let mut final_vec = core::Vector::<Mat>::new();
    final_vec.push(top_row);
    final_vec.push(bottom_row);
    core::vconcat(&final_vec, &mut final_vis)?;

    Ok(final_vis)
}

#[derive(Debug, Serialize)]
struct Summary {
    video1: VideoInfo,
    video2: VideoInfo,
    comparison: ComparisonInfo,
    statistics: StatisticsInfo,
    warnings: Vec<String>,
    different_groups: Vec<GroupSummary>,
}

#[derive(Debug, Serialize)]
struct VideoInfo {
    path: String,
    frames: i32,
    fps: f64,
    resolution: String,
}

#[derive(Debug, Serialize)]
struct ThresholdsUsed {
    mse_threshold: f64,
    ssim_threshold: f64,
    diff_pct_threshold: f64,
    edge_diff_pct_threshold: f64,
    blob_diff_pixels_threshold: f64,
    local_ssim_threshold: f64,
}

#[derive(Debug, Serialize)]
struct ComparisonInfo {
    frames_compared: usize,
    frame_group_size: usize,
    total_groups: usize,
    significant_groups: usize,
    significant_frames: usize,
    percentage_different: f64,
    thresholds_used: ThresholdsUsed,
}

#[derive(Debug, Serialize)]
struct StatBlock {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
}

#[derive(Debug, Serialize)]
struct StatisticsInfo {
    mse: StatBlock,
    ssim: StatBlock,
    diff_percentage: StatBlock,
    edge_diff_percentage: StatBlock,
    blob_diff_pixels: StatBlock,
    min_local_ssim: StatBlock,
}

#[derive(Debug, Serialize)]
struct GroupSummary {
    frames: String,
    timestamp: String,
    avg_mse: f64,
    avg_ssim: f64,
    avg_diff_pct: f64,
    avg_edge_diff_pct: f64,
    avg_blob_diff_pixels: f64,
    avg_min_local_ssim: f64,
    individual_frames: Vec<FrameSummary>,
}

#[derive(Debug, Serialize)]
struct FrameSummary {
    frame: i32,
    mse: f64,
    ssim: f64,
    diff_pct: f64,
    edge_diff_pct: f64,
    blob_diff_pixels: f64,
    min_local_ssim: f64,
}

fn build_summary(
    video1_meta: &VideoMeta,
    video2_meta: &VideoMeta,
    frame_group_size: usize,
    thresholds: Thresholds,
    frame_area: f64,
    warnings: Vec<String>,
    all_differences: &[FrameDifference],
    significant_groups: &[FrameGroupDifference],
) -> Summary {
    let mse_values: Vec<f64> = all_differences.iter().map(|d| d.mse).collect();
    let ssim_values: Vec<f64> = all_differences.iter().map(|d| d.ssim).collect();
    let diff_values: Vec<f64> = all_differences.iter().map(|d| d.diff_percentage).collect();
    let edge_diff_values: Vec<f64> = all_differences
        .iter()
        .map(|d| d.edge_diff_percentage)
        .collect();
    let blob_diff_values: Vec<f64> = all_differences.iter().map(|d| d.blob_diff_pixels).collect();
    let min_local_ssim_values: Vec<f64> =
        all_differences.iter().map(|d| d.min_local_ssim).collect();

    let total_significant_frames: usize = significant_groups
        .iter()
        .map(|g| g.individual_frames.len())
        .sum();

    let frames_compared = all_differences.len();
    let total_groups = if frame_group_size == 0 {
        0
    } else {
        frames_compared.div_ceil(frame_group_size)
    };
    let percentage_different = if frames_compared > 0 {
        (total_significant_frames as f64 / frames_compared as f64) * 100.0
    } else {
        0.0
    };

    Summary {
        video1: VideoInfo {
            path: video1_meta.path.clone(),
            frames: video1_meta.total_frames,
            fps: video1_meta.fps,
            resolution: format!("{}x{}", video1_meta.width, video1_meta.height),
        },
        video2: VideoInfo {
            path: video2_meta.path.clone(),
            frames: video2_meta.total_frames,
            fps: video2_meta.fps,
            resolution: format!("{}x{}", video2_meta.width, video2_meta.height),
        },
        comparison: ComparisonInfo {
            frames_compared,
            frame_group_size,
            total_groups,
            significant_groups: significant_groups.len(),
            significant_frames: total_significant_frames,
            percentage_different,
            thresholds_used: ThresholdsUsed {
                mse_threshold: thresholds.mse,
                ssim_threshold: thresholds.ssim,
                diff_pct_threshold: thresholds.diff_pct,
                edge_diff_pct_threshold: thresholds.edge_diff_pct,
                blob_diff_pixels_threshold: (thresholds.blob_diff_pct / 100.0) * frame_area,
                local_ssim_threshold: thresholds.local_ssim,
            },
        },
        statistics: StatisticsInfo {
            mse: build_stat_block(&mse_values),
            ssim: build_stat_block(&ssim_values),
            diff_percentage: build_stat_block(&diff_values),
            edge_diff_percentage: build_stat_block(&edge_diff_values),
            blob_diff_pixels: build_stat_block(&blob_diff_values),
            min_local_ssim: build_stat_block(&min_local_ssim_values),
        },
        warnings,
        different_groups: significant_groups
            .iter()
            .map(|g| GroupSummary {
                frames: if g.start_frame == g.end_frame {
                    g.start_frame.to_string()
                } else {
                    format!("{}-{}", g.start_frame, g.end_frame)
                },
                timestamp: if g.start_frame == g.end_frame {
                    format!("{:.2}s", g.start_timestamp_sec)
                } else {
                    format!("{:.2}s-{:.2}s", g.start_timestamp_sec, g.end_timestamp_sec)
                },
                avg_mse: round_n(g.avg_mse, 2),
                avg_ssim: round_n(g.avg_ssim, 4),
                avg_diff_pct: round_n(g.avg_diff_percentage, 2),
                avg_edge_diff_pct: round_n(g.avg_edge_diff_percentage, 2),
                avg_blob_diff_pixels: round_n(g.avg_blob_diff_pixels, 0),
                avg_min_local_ssim: round_n(g.avg_min_local_ssim, 4),
                individual_frames: g
                    .individual_frames
                    .iter()
                    .map(|f| FrameSummary {
                        frame: f.frame_number,
                        mse: round_n(f.mse, 2),
                        ssim: round_n(f.ssim, 4),
                        diff_pct: round_n(f.diff_percentage, 2),
                        edge_diff_pct: round_n(f.edge_diff_percentage, 2),
                        blob_diff_pixels: round_n(f.blob_diff_pixels, 0),
                        min_local_ssim: round_n(f.min_local_ssim, 4),
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn build_stat_block(values: &[f64]) -> StatBlock {
    if values.is_empty() {
        return StatBlock {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
        };
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|v| {
            let diff = v - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    let std = variance.sqrt();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    StatBlock {
        mean,
        std,
        min,
        max,
    }
}

fn round_n(value: f64, decimals: u32) -> f64 {
    let factor = 10f64.powi(decimals as i32);
    (value * factor).round() / factor
}
