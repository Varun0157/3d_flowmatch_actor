DATA_PATH=peract2_raw/
ZARR_PATH=zarr_datasets/peract2_dense/

# Download raw PerAct2 episodes (includes full joint-space trajectory data)
python scripts/rlbench/download_peract2.py --root ${DATA_PATH}

# Download the test seeds
CURR_DIR=$(pwd)
cd ${DATA_PATH}
wget https://huggingface.co/katefgroup/3d_flowmatch_actor/resolve/main/peract2_test.zip
unzip peract2_test.zip
rm peract2_test.zip
cd "$CURR_DIR"

# Convert raw episodes to dense zarr (samples actual simulator frames, no interpolation)
python data_processing/peract2_to_zarr_dense.py \
    --root ${DATA_PATH} \
    --tgt ${ZARR_PATH} \
    --max_steps 15
# You can safely delete the raw train/val data now, not test
