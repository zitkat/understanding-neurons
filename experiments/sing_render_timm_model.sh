#!/bin/bash
#PBS -q gpu
#PBS -l walltime=16:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_ssd=30gb:gpu_cap=cuda75
#PBS -j oe
#PBS -o /storage/plzen1/home/zitkat/
#PBS -m ae
#trap 'clean_scratch' TERM EXIT

model_name=mobilenetv3_rw
weights=pretrained

sing_image=lucent_torch_21.03-py3.sif


# -- tested by:
##$ qsub -I -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_ssd=10gb:gpu_cap=cuda75 -l walltime=1:00:00 -q gpu

cp -r /storage/plzen1/home/zitkat/understanding-critical-neurons/ "$SCRATCHDIR" || exit $LINENO

WORK_PATH=$SCRATCHDIR/understanding-critical-neurons/
DATA_PATH=$SCRATCHDIR/understanding-critical-neurons/data
OUTPUT_PATH=$SCRATCHDIR/understanding-critical-neurons/output

cd "$WORK_PATH" || exit $LINENO

cp "$DATA_PATH"/"$sing_image" . || exit $LINENO

#export SINGULARITY_CACHEDIR=/storage/plzen1/home/zitkat/.singularity_cache
#export SINGULARITY_LOCALCACHEDIR="$SCRATCHDIR"
#mkdir "$SCRATHDIR"/tmp && export SINGULARITY_TMPDIR="$SCRATCHDIR"/tmp
singularity exec "$sing_image" \
   python -m pip install  git+git://github.com/zitkat/lucent.git@mymaster#egg=torch_lucent || exit $LINENO

today=$(date +%Y%m%d%H%M)
singularity exec --nv -B "$SCRATCHDIR"  "$sing_image" \
  python render_timm_model.py "$model_name" \
                            --model-weights "$weights" \
                            --layers "$DATA_PATH"/renders/mobilenetv3_rw_pretrained/todolayers.list \
                            -sv v1 \
                            --settings-file settings.csv\
                            --output "$OUTPUT_PATH"/renders \
                            --hide-progress > "$DATA_PATH"/"$today"_"$model_name"_render_timm_model.log

cp "$WORK_PATH"/sing_render_timm_model.sh "$OUTPUT_PATH"/"$today"_"$model_name"_sing_render_timm_model.sh
cp -ru "$OUTPUT_PATH"/* "$DATA_PATH"



