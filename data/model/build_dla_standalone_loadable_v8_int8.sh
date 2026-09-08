#!/bin/bash
# Build the DLA INT8 loadable for an Ultralytics-family model (v8 head style,
# e.g. yolov8s; adapt names for v5su). Inputs produced on the training/GPU
# machine by the ultralytics toolkit (see docs/ultralytics-support.zh-CN.md):
#   python scripts/qat_dla.py quantize weights/yolov8s.pt --calib-dir <imgs> --imgsz 672 --qat qat.pt
#   python scripts/export_dla.py --weights qat.pt --qat --size 672 --save yolov8s_qat.onnx
#   qdq_translator.py --input_onnx_models=yolov8s_qat.onnx --infer_concat_scales --infer_mul_scales
# Deploying requires: HEAD_STYLE=v8 for the matx rebuild and make, plus
# INPUT_SCALE=0.007874015718698502 (the cache's images: entry) in make.
mkdir -p data/loadable
TRTEXEC=/usr/src/tensorrt/bin/trtexec
# TensorRT 10.x refuses the TRT-8600 cache header; rewrite to the local TRT version.
TRT_VERSION=$(${TRTEXEC} --version 2>&1 | grep -oE 'TensorRT v[0-9]+' | head -1 | grep -oE '[0-9]+')
CALIB_CACHE=data/loadable/qat2ptq_v8.cache
sed "1s/^TRT-[0-9]*/TRT-${TRT_VERSION}/" data/model/yolov8s_qat_precision_config_calib.cache > ${CALIB_CACHE}

# FP16 fallback set: the final box/cls head convs per level + the head concats
# feeding the raw outputs (s8/s16/s32 carry no INT8 scales).
${TRTEXEC} --minShapes=images:1x3x672x672 --maxShapes=images:1x3x672x672 --optShapes=images:1x3x672x672 --shapes=images:1x3x672x672 \
    --onnx=data/model/yolov8s_qat_noqdq.onnx --useDLACore=0 --buildDLAStandalone \
    --saveEngine=data/loadable/yolov8s.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --int8 --fp16 \
    --calib=${CALIB_CACHE} --precisionConstraints=obey \
    --layerPrecisions="/model.22/cv2.0/cv2.0.2/Conv":fp16,"/model.22/cv2.1/cv2.1.2/Conv":fp16,"/model.22/cv2.2/cv2.2.2/Conv":fp16,"/model.22/cv3.0/cv3.0.2/Conv":fp16,"/model.22/cv3.1/cv3.1.2/Conv":fp16,"/model.22/cv3.2/cv3.2.2/Conv":fp16,"/model.22/Concat":fp16,"/model.22/Concat_1":fp16,"/model.22/Concat_2":fp16
