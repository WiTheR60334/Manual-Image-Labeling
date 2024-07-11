def format_results(boxes, scores, image_id, cat_id):
    results = []
    for box, score in zip(boxes, scores):
        # Create a dictionary for each detected object
        r = {
            "image_id": image_id,       # Image ID to associate with the detection
            "category_id": cat_id,      # Category ID (class label) of the detected object
            "bbox": [float(i) for i in box],  # Bounding box coordinates [xmin, ymin, width, height]
            "score": float(score),      # Confidence score of the detection
        }
        results.append(r)  # Append the formatted result dictionary to the results list
    return results  # Return the list of formatted detection results