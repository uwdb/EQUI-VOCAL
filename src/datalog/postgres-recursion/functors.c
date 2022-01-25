#include "postgres.h"
#include <stdint.h>
#include "fmgr.h"

PG_FUNCTION_INFO_V1(get_iou);
PG_MODULE_MAGIC;

Datum get_iou(PG_FUNCTION_ARGS) {

    float x1 = PG_GETARG_FLOAT8(0);
    float y1 = PG_GETARG_FLOAT8(1);
    float x2 = PG_GETARG_FLOAT8(2);
    float y2 = PG_GETARG_FLOAT8(3);
    float x1p = PG_GETARG_FLOAT8(4);
    float y1p = PG_GETARG_FLOAT8(5);
    float x2p = PG_GETARG_FLOAT8(6);
    float y2p = PG_GETARG_FLOAT8(7);

    // determine the coordinates of the intersection rectangle
    float x_left = (x1 > x1p) ? x1 : x1p;
    float y_top = (y1 > y1p) ? y1 : y1p;
    float x_right = (x2 < x2p) ? x2 : x2p;
    float y_bottom = (y2 < y2p)? y2 : y2p;

    if (x_right < x_left || y_bottom < y_top) {
        PG_RETURN_FLOAT8(0.0);
    }
    // The intersection of two axis-aligned bounding boxes is always an
    // axis-aligned bounding box
    float intersection_area = (x_right - x_left) * (y_bottom - y_top);

    // compute the area of both AABBs
    float bb1_area = (x2 - x1) * (y2 - y1);
    float bb2_area = (x2p - x1p) * (y2p - y1p);

    // compute the intersection over union by taking the intersection
    // area and dividing it by the sum of prediction + ground-truth
    // areas - the interesection area
    float iou = intersection_area / (bb1_area + bb2_area - intersection_area);
    PG_RETURN_FLOAT8(iou);
}