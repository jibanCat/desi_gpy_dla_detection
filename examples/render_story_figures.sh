#!/bin/bash
# Render the inline-figure set for the per-mock story docs.
# Each invocation produces one PNG in docs/story_figures/.
# Sequenced (one at a time) so it shares CPUs nicely with running jobs;
# total wall time ~12 figures × ~1 min = ~12 min.
set -u
cd /home/mfho/desi_gpy_dla_detection
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpdla
export LD_LIBRARY_PATH="$HOME/.local/usr/local/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p docs/story_figures

run_one() {
    local mock="$1" tid="$2" spec="$3" zcat="$4" tz="$5" tn="$6" out="$7"
    # Look up truth + BAL catalogs from the spec path's mock directory.
    local mock_dir
    mock_dir="$(dirname $(dirname $(dirname "$spec")))"  # spectra-16/X/Y → mock dir
    local truth_cat="$mock_dir/hcd_truth_cat.fits"
    [ -f "$truth_cat" ] || truth_cat="$mock_dir/dla_cat.fits"  # london naming
    local bal_cat="$mock_dir/bal_cat.fits"
    echo "=== ${mock} TID=${tid} → ${out} ==="
    python -u examples/plot_one_spectrum_with_fit.py \
        --mock "$mock" --target-id "$tid" \
        --spec "$spec" --zcat "$zcat" \
        --truth-z "$tz" --truth-log-nhi "$tn" \
        --truth-catalog "$truth_cat" \
        --bal-catalog "$bal_cat" \
        --out-png "$out" 2>&1 | grep -E "^\[|τ_factor|Voigt|saved" | tail -10
}

# 2lpt (Phase B 5k results — actual production-bayes biases)
M2LPT_BASE=/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/lyacolore_2lpt/qq_desi_y3/v2.8.5/mock-0/loa-124
ZCAT_2LPT=$M2LPT_BASE/zcat.fits

run_one 2lpt 120046865 \
    "$M2LPT_BASE/spectra-16/7/789/spectra-16-789.fits" "$ZCAT_2LPT" \
    2.7730 21.263 docs/story_figures/2lpt_01_canonical_dla.png

run_one 2lpt 260080167 \
    "$M2LPT_BASE/spectra-16/17/1704/spectra-16-1704.fits" "$ZCAT_2LPT" \
    2.5517 21.055 docs/story_figures/2lpt_02_strong_dla_closes.png

run_one 2lpt 60167537 \
    "$M2LPT_BASE/spectra-16/3/396/spectra-16-396.fits" "$ZCAT_2LPT" \
    2.5617 20.618 docs/story_figures/2lpt_03_mid_dla.png

run_one 2lpt 88448 \
    "$M2LPT_BASE/spectra-16/0/42/spectra-16-42.fits" "$ZCAT_2LPT" \
    -1.0 -1.0 docs/story_figures/2lpt_04_false_positive_rescue.png

# london (n=54 picker results, diagnostic-recipe biases)
MLON_BASE=/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/london/qq_desi_y3/v5.9.5/mock-0/jura-124
ZCAT_LON=$MLON_BASE/zcat.fits

run_one london 100302972 \
    "$MLON_BASE/spectra-16/17/1707/spectra-16-1707.fits" "$ZCAT_LON" \
    2.218 20.93 docs/story_figures/london_01_dla_modest_bias.png

run_one london 140016836 \
    "$MLON_BASE/spectra-16/22/2287/spectra-16-2287.fits" "$ZCAT_LON" \
    2.069 20.96 docs/story_figures/london_02_dla_large_bias.png

run_one london 180258638 \
    "$MLON_BASE/spectra-16/5/543/spectra-16-543.fits" "$ZCAT_LON" \
    2.808 20.33 docs/story_figures/london_03_marginal_dla.png

# saclay (n=54 picker results)
MSAC_BASE=/nfs/turbo/lsa-cavestru/mfho/DESI/mocks/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124
ZCAT_SAC=$MSAC_BASE/zcat.fits

run_one saclay 1377001320 \
    "$MSAC_BASE/spectra-16/2/274/spectra-16-274.fits" "$ZCAT_SAC" \
    2.487 20.88 docs/story_figures/saclay_01_dla_clean_close.png

run_one saclay 6388000890 \
    "$MSAC_BASE/spectra-16/10/1081/spectra-16-1081.fits" "$ZCAT_SAC" \
    2.078 21.65 docs/story_figures/saclay_02_strongest_dla.png

run_one saclay 2092000495 \
    "$MSAC_BASE/spectra-16/17/1719/spectra-16-1719.fits" "$ZCAT_SAC" \
    1.874 20.96 docs/story_figures/saclay_03_dla_persistent_bias.png

echo "ALL FIGURES DONE"
ls -la docs/story_figures/
