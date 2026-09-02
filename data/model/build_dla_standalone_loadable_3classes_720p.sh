#!/bin/bash
# Build the DLA loadables for the 3-class custom model at 720p (736x1280,
# HxW — both multiples of 32). Same precision set as the v1 672 script.
# The 720p ONNX inputs come from (see yolov5_dla docs for the two export flavors):
#   QAT:   python scripts/qat.py export qat_1280.pt --size=736x1280 --save=yolov5_3clases_qat_720p.onnx --dynamic --noanchor
#   FP32:  python scripts/qat.py export runs/train/3classes_1280/weights/best.pt --size=736x1280 \
#              --save=yolov5_3clases_fp32_720p.onnx --dynamic --noanchor --noqadd
# then translate the QAT one with qdq_translator (--infer_concat_scales --infer_mul_scales).
# Deploying requires: NUM_CLASSES=3 INPUT_H=736 INPUT_W=1280 for both the matx
# rebuild and make (INPUT must match the export size exactly).
mkdir -p data/loadable
TRTEXEC=/usr/src/tensorrt/bin/trtexec
# TensorRT 10.x refuses the TRT-8600 cache header; rewrite it to the local TRT version.
TRT_VERSION=$(${TRTEXEC} --version 2>&1 | grep -oE 'TensorRT v[0-9]+' | head -1 | grep -oE '[0-9]+')
CALIB_CACHE=data/loadable/qat2ptq_3classes_720p.cache
sed "1s/^TRT-[0-9]*/TRT-${TRT_VERSION}/" data/model/yolov5_3clases_qat_720p_precision_config_calib.cache > ${CALIB_CACHE}
SHAPES="images:1x3x736x1280"

# FP16 loadable
${TRTEXEC} --onnx=data/model/yolov5_3clases_fp32_720p.onnx --fp16 \
    --saveEngine=data/loadable/yolov5_3classes_720p.fp16.fp16chw16in.fp16chw16out.standalone.bin \
    --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 --buildDLAStandalone --useDLACore=0

# INT8 loadable (3 head convs in FP16)
${TRTEXEC} --minShapes=${SHAPES} --maxShapes=${SHAPES} --optShapes=${SHAPES} --shapes=${SHAPES} \
    --onnx=data/model/yolov5_3clases_qat_720p_noqdq.onnx --useDLACore=0 --buildDLAStandalone \
    --saveEngine=data/loadable/yolov5_3classes_720p.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --int8 --fp16 \
    --calib=${CALIB_CACHE} --precisionConstraints=obey \
    --layerPrecisions="/model.24/m.0/Conv":fp16,"/model.24/m.1/Conv":fp16,"/model.24/m.2/Conv":fp16
