#!/bin/bash
# Build the DLA INT8 loadable from the 3-class custom QAT model
# (yolov5_3clases_qat.onnx translated by qdq_translator → *_noqdq.onnx + cache).
# Derived from build_dla_standalone_loadable.sh — INT8 part only.
# NOTE: deploying this model also requires the nc=3 C++ adaptations
# (see CLAUDE.md "Custom-Dataset (non-COCO) Workflow").
mkdir -p data/loadable
TRTEXEC=/usr/src/tensorrt/bin/trtexec
# TensorRT 10.x refuses the TRT-8600 cache header ("Calibration table does not
# match calibrator algorithm type") and then fails trying to recalibrate with
# no data. Rewrite the header to the locally installed TRT version.
TRT_VERSION=$(${TRTEXEC} --version 2>&1 | grep -oE 'TensorRT v[0-9]+' | head -1 | grep -oE '[0-9]+')
CALIB_CACHE=data/loadable/qat2ptq_3classes.cache
sed "1s/^TRT-[0-9]*/TRT-${TRT_VERSION}/" data/model/yolov5_3clases_qat_precision_config_calib.cache > ${CALIB_CACHE}
${TRTEXEC} --minShapes=images:1x3x672x672 --maxShapes=images:1x3x672x672 --optShapes=images:1x3x672x672 --shapes=images:1x3x672x672 \
    --onnx=data/model/yolov5_3clases_qat_noqdq.onnx --useDLACore=0 --buildDLAStandalone \
    --saveEngine=data/loadable/yolov5_3classes.int8.int8hwc4in.fp16chw16out.standalone.bin \
    --inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --int8 --fp16 \
    --calib=${CALIB_CACHE} --precisionConstraints=obey \
    --layerPrecisions="/model.24/m.0/Conv":fp16,"/model.24/m.1/Conv":fp16,"/model.24/m.2/Conv":fp16
