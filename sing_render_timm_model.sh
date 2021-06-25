#!/bin/bash
#PBS -q gpu
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_ssd=30gb:cluster=adan
#PBS -j oe
#PBS -o /storage/plzen1/home/zitkat/
#PBS -m ae
trap 'clean_scratch' TERM EXIT

model_name=mobilenetv3_rw
weights=pretrained

sing_image=tbt_torch_21.03-py3.sif


# -- tested by:
##$ qsub -I -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_ssd=10gb -l walltime=0:30:00 -q gpu

cp -r /storage/plzen1/home/zitkat/understanding-critical-neurons/ "$SCRATCHDIR" || exit $LINENO

DATA_PATH=$SCRATCHDIR/understanding-critical-neurons/data
WORK_PATH=$SCRATCHDIR/understanding-critical-neurons/
OUTPUT_PATH=$SCRATCHDIR/understanding-critical-neurons/output

cd "$WORK_PATH" || exit $LINENO

cp "$DATA_PATH"/"$sing_image" . || exit $LINENO

#export SINGULARITY_CACHEDIR=/storage/plzen1/home/zitkat/sing_cache
#export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR
#export SINGULARITY_TMPDIR=""
singularity exec "$sing_image" \
   python -m pip install  git+git://github.com/zitkat/lucent.git@mymaster#egg=torch_lucent || exit $LINENO

today=$(date +%Y%m%d%H%M)
singularity exec --nv -B "$SCRATCHDIR"  "$sing_image" \
  python render_timm_model.py "$model_name" \
                            --model-weights "$weights" \
                            -sv v1 \
                            --settings-file settings.csv\
                            --output "$OUTPUT_PATH"/renders \
                            --hide-progress > "$DATA_PATH"/"$today".log
cp -ru "$OUTPUT_PATH" "$DATA_PATH" || export CLEAN_SCRATCH=False



