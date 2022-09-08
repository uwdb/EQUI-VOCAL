#include "postgres.h"
#include <stdint.h>
#include <math.h>
#include "fmgr.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(get_iou);

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

PG_FUNCTION_INFO_V1(near);

Datum near(PG_FUNCTION_ARGS) {

    float x1 = PG_GETARG_FLOAT8(0);
    float y1 = PG_GETARG_FLOAT8(1);
    float x2 = PG_GETARG_FLOAT8(2);
    float y2 = PG_GETARG_FLOAT8(3);
    float x3 = PG_GETARG_FLOAT8(4);
    float y3 = PG_GETARG_FLOAT8(5);
    float x4 = PG_GETARG_FLOAT8(6);
    float y4 = PG_GETARG_FLOAT8(7);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;
    float cx2 = (x3 + x4) / 2;
    float cy2 = (y3 + y4) / 2;

    float distance = sqrt(pow(cx1 - cx2, 2.0) + pow(cy1 - cy2, 2.0)) / ((x2 - x1 + x4 - x3) / 2);

    PG_RETURN_BOOL(distance < 1.05);
}

PG_FUNCTION_INFO_V1(far);

Datum far(PG_FUNCTION_ARGS) {

    float x1 = PG_GETARG_FLOAT8(0);
    float y1 = PG_GETARG_FLOAT8(1);
    float x2 = PG_GETARG_FLOAT8(2);
    float y2 = PG_GETARG_FLOAT8(3);
    float x3 = PG_GETARG_FLOAT8(4);
    float y3 = PG_GETARG_FLOAT8(5);
    float x4 = PG_GETARG_FLOAT8(6);
    float y4 = PG_GETARG_FLOAT8(7);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;
    float cx2 = (x3 + x4) / 2;
    float cy2 = (y3 + y4) / 2;

    float distance = sqrt(pow(cx1 - cx2, 2.0) + pow(cy1 - cy2, 2.0)) / ((x2 - x1 + x4 - x3) / 2);

    PG_RETURN_BOOL(distance > 0.9);
}

PG_FUNCTION_INFO_V1(left_of);

Datum left_of(PG_FUNCTION_ARGS) {

    float x1 = PG_GETARG_FLOAT8(0);
    float y1 = PG_GETARG_FLOAT8(1);
    float x2 = PG_GETARG_FLOAT8(2);
    float y2 = PG_GETARG_FLOAT8(3);
    float x3 = PG_GETARG_FLOAT8(4);
    float y3 = PG_GETARG_FLOAT8(5);
    float x4 = PG_GETARG_FLOAT8(6);
    float y4 = PG_GETARG_FLOAT8(7);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;
    float cx2 = (x3 + x4) / 2;
    float cy2 = (y3 + y4) / 2;

    PG_RETURN_BOOL(cx1 < cx2);
}

PG_FUNCTION_INFO_V1(behind);

Datum behind(PG_FUNCTION_ARGS) {

    float x1 = PG_GETARG_FLOAT8(0);
    float y1 = PG_GETARG_FLOAT8(1);
    float x2 = PG_GETARG_FLOAT8(2);
    float y2 = PG_GETARG_FLOAT8(3);
    float x3 = PG_GETARG_FLOAT8(4);
    float y3 = PG_GETARG_FLOAT8(5);
    float x4 = PG_GETARG_FLOAT8(6);
    float y4 = PG_GETARG_FLOAT8(7);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;
    float cx2 = (x3 + x4) / 2;
    float cy2 = (y3 + y4) / 2;

    PG_RETURN_BOOL(cy1 < cy2);
}

PG_FUNCTION_INFO_V1(right_quadrant);

Datum right_quadrant(PG_FUNCTION_ARGS) {

    float x1 = PG_GETARG_FLOAT8(0);
    float y1 = PG_GETARG_FLOAT8(1);
    float x2 = PG_GETARG_FLOAT8(2);
    float y2 = PG_GETARG_FLOAT8(3);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;

    PG_RETURN_BOOL(cx1 >= 240 && cx1 <= 480);
}

PG_FUNCTION_INFO_V1(top_quadrant);

Datum top_quadrant(PG_FUNCTION_ARGS) {

    float x1 = PG_GETARG_FLOAT8(0);
    float y1 = PG_GETARG_FLOAT8(1);
    float x2 = PG_GETARG_FLOAT8(2);
    float y2 = PG_GETARG_FLOAT8(3);

    float cx1 = (x1 + x2) / 2;
    float cy1 = (y1 + y2) / 2;

    PG_RETURN_BOOL(cy1 >= 0 && cy1 < 160);
}