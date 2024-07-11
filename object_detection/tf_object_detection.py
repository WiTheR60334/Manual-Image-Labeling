import numpy as np
import tensorflow as tf

class ObjectDetector(object):
    def __init__(self, graph_path, score_threshold, objIds):
        # Initialize ObjectDetector instance
        self.detection_graph = self._load_graph(graph_path)  # Load TensorFlow detection graph
        self.input_tensor, self.tensor_dict = self._get_input_output_tensors(self.detection_graph)  # Get input and output tensors from the graph
        self.sess = tf.Session(graph=self.detection_graph)  # Create TensorFlow session
        self.score_threshold = score_threshold  # Threshold for object detection confidence scores
        self.objIds = objIds  # List of object IDs to detect

    def _load_graph(self, graph_path):
        # Load TensorFlow graph from a protobuf file (.pb)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _get_input_output_tensors(self, graph):
        # Retrieve input and output tensors from the TensorFlow graph
        with graph.as_default():
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        return image_tensor, tensor_dict

    def _post_process(self, output_dict, im_width, im_height):
        # Post-process the output dictionary to filter, scale, and convert bounding boxes
        # Convert detection results to appropriate format
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        
        # Filter out boxes with low confidences based on the threshold
        boxes = output_dict["detection_boxes"]
        class_indices = output_dict["detection_classes"]
        scores = output_dict["detection_scores"]
        mask = scores > self.score_threshold
        boxes = boxes[mask, :]
        class_indices = class_indices[mask]
        scores = scores[mask]
        
        # Only keep classes listed in objIds (if specified)
        if self.objIds is not None:
            mask = np.isin(class_indices, self.objIds)
            boxes = boxes[mask]
            class_indices = class_indices[mask]
            scores = scores[mask]
        
        # Scale the box dimensions to match the input image size
        boxes[:, [0, 2]] *= im_height
        boxes[:, [1, 3]] *= im_width
        
        # Convert from (ymin, xmin, ymax, xmax) to (xmin, ymin, width, height) format
        new_boxes = np.empty_like(boxes)
        new_boxes[:, 0] = boxes[:, 1]  # xmin
        new_boxes[:, 1] = boxes[:, 0]  # ymin
        new_boxes[:, 2] = boxes[:, 3] - boxes[:, 1]  # width
        new_boxes[:, 3] = boxes[:, 2] - boxes[:, 0]  # height
        boxes = new_boxes
        
        return boxes, scores, class_indices

    def detect(self, im):
        # Perform object detection on input image (assumed to be in RGB color space)
        height, width = im.shape[:2]
        
        # Run the TensorFlow session to perform inference
        output_dict = self.sess.run(self.tensor_dict,
                                    feed_dict={self.input_tensor: np.expand_dims(im, 0)})
        
        # Post-process the detection results
        boxes, scores, class_indices = self._post_process(output_dict, width, height)
        
        return boxes, scores, class_indices