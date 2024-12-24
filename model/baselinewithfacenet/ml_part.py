from util import Get_normal_bbox
from detection import mtcnn_detection, mtcnn_get_embeddings, mtcnn_recognition
from main import yolov8_detection
def Detection(img, args, model_args):
    # Detection using selected model
    device = model_args['Device']

    if args['DETECTOR'] == 'mtcnn':
        bboxes = mtcnn_detection(model_args['Detection'], img, device)
    elif args['DETECTOR'] == 'yolov8':
        bboxes = yolov8_detection(model_args['Detection'], img)
    else:
        raise ValueError(f"Unknown detector: {args['DETECTOR']}")

    if args['DEBUG_MODE']:
        print(f"Detected bboxes: {bboxes}")
    # Normalize bounding boxes to stay within image bounds
    if bboxes is not None:
        bboxes = Get_normal_bbox(img.shape, bboxes)

    return bboxes


def Recognition(img, bboxes, args, model_args):
    device = model_args['Device']

    faces, unknown_embeddings = mtcnn_get_embeddings(
        model_args['Mtcnn'], 
        model_args['Recognition'], 
        img, bboxes, device
    )

    face_ids, result_probs = mtcnn_recognition(
        model_args['Face_db'], 
        unknown_embeddings, 
        args['RECOG_THRESHOLD'], 
        similarity_metric=args['SIMILARITY_METRIC']  # Pass similarity metric
    )

    return face_ids

