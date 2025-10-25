# Datasets & Provenance

I do **not** commit full third-party videos. Instead:
- **DCSASS** (Kaggle: `mateohervas/dcsass-dataset`, CC-BY-NC-SA-4.0)  
  Used: `Shoplifting/**/*.mp4`
- **UCSD Anomaly Detection v1p2** (Kaggle mirror: `karthiknm1/ucsd-anomaly-detection-dataset`)  
  Used: `UCSDped1/Test/*` (first 16 folders) converted via `tools/ucsd_frames_to_mp4.py`.

## Recreate
\`\`\`bash
bash tools/recreate_data.sh
\`\`\`

## Included in repo
- \`labels.csv\`
- \`pose_out_sample/\` (100 positive + 93 negative tracks)
- \`output/alert_log.csv\`, \`output/alert_events.json\`
- \`output/demos/*.mp4\` (event demo clips)
- \`runs/lstm_skeleton_best.pt\` (Git LFS)
