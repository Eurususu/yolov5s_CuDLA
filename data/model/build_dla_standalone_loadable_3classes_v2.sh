#!/bin/bash
# Build the DLA loadables for the 3-class custom model (v2 precision set).
# Same as build_dla_standalone_loadable_3classes.sh except the INT8 loadable
# falls back MORE layers to FP16 — the 3 head convs plus the model.23/cv3
# branch (conv/Sigmoid/Mul) — trading speed for accuracy (on the 80-class
# COCO model this raised mAP 37.1 → 37.3 at 4.0 → 4.46 ms; expect a similar
# trade here). The INT8 output gets a _v2 name so both variants can coexist;
# the FP16 loadable is identical to the v1 script's and overwrites it.
# Deploying requires the matching app build: make NUM_CLASSES=3
mkdir -p data/loadable
TRTEXEC=/usr/src/tensorrt/bin/trtexec
# TensorRT 10.x refuses the TRT-8600 cache header; rewrite it to the local TRT version.
TRT_VERSION=$(${TRTEXEC} --version 2>&1 | grep -oE 'TensorRT v[0-9]+' | head -1 | grep -oE '[0-9]+')
CALIB_CACHE=data/loadable/qat2ptq_3classes.cache
sed "1s/^TRT-[0-9]*/TRT-${TRT_VERSION}/" data/model/yolov5_3clases_qat_precision_config_calib.cache > ${CALIB_CACHE}

# FP16 loadable (same artifact as the v1 script)
${TRTEXEC} --onnx=data/model/yolov5_3clases_fp32_trimmed.onnx --fp16 \
    --saveEngine=data/loadable/yolov5_3classes.fp16.fp16chw16in.fp16chw16out.standalone.bin \
    --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 --buildDLAStandalone --useDLACore=0

# INT8 loadable, v2 precision set (3 head convs + model.23/cv3 branch in FP16)
${TRTEXEC} --minShapes=images:1x3x672x672 --maxShapes=images:1x3x672x672 --optShapes=images:1x3x672x672 --shapes=images:1x3x672x672 \
    --onnx=data/model/yolov5_3clases_qat_noqdq.onnx --useDLACore=0 --buildDLAStandalone \
    --saveEngine=data/loadable/yolov5_3classes_v2.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --int8 --fp16 \
    --calib=${CALIB_CACHE} --precisionConstraints=obey \
    --layerPrecisions="/model.24/m.0/Conv":fp16,"/model.24/m.1/Conv":fp16,"/model.24/m.2/Conv":fp16,"/model.23/cv3/conv/Conv":fp16,"/model.23/cv3/act/Sigmoid":fp16,"/model.23/cv3/act/Mul":fp16
