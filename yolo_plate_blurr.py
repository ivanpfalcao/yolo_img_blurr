from ultralytics import YOLO
import os
import logging
import cv2

logging.basicConfig(level=logging.INFO)

class ImageBlurr():
    def __init__(self, yolo_model: str = 'yolo11n.pt'):
        self.yolo_model = yolo_model
        self.model: YOLO = YOLO(self.yolo_model)
        logging.info(f"Yolo model name: {yolo_model}")

    def model_train(self, yaml_data_path, epochs = 100, imgsz = 640):
        self.model.train(data=yaml_data_path, epochs = epochs, imgsz = imgsz)

    def predict(
            self
            , image_to_blurr
            , project: str = 'detect'
            , name: str = 'test'
            , save: bool = True
            , exist_ok: bool = True):
        
        result = self.model.predict(
            source=image_to_blurr
            , save=save
            , show=False
            , project=project
            , name=name
            , exist_ok=exist_ok
        ) 

        return result
    
    def blurr_image(self, input_image, output_img_path):

        file_name = os.path.basename(input_image)
        output_path = os.path.join(output_img_path, file_name)
        logging.info(f'Input Image: {input_image}')
        logging.info(f'Output Image: {output_path}')


        results = self.predict(
            image_to_blurr=input_image
            , save=False
        )   

        image = cv2.imread(input_image)

        # Iterate through the results to get bounding boxes
        for r in results:
            for box in r.boxes:
                # Get the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                # Define the region of interest (ROI)
                roi = image[y1:y2, x1:x2]
                
                # Apply a Gaussian blur to the ROI
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 0)
                
                # Replace the original ROI with the blurred ROI
                image[y1:y2, x1:x2] = blurred_roi

        # Save the final blurred image
        cv2.imwrite(output_path, image)

        # Display the image (optional)
        #cv2.imshow('Blurred Image', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        logging.info(f"Blurred image saved to {output_path}")        
    
class PlateBlurr():
    def __init__(self, yolo_train_model: str = None):
        current_file = os.path.abspath(__file__)
        self.current_file_path = os.path.dirname(current_file) 

        if yolo_train_model is None:
            self.img_blurr = ImageBlurr()
        else:
            self.img_blurr = ImageBlurr(yolo_train_model)
        

    def model_train(self):
        self.img_blurr.model_train(
            yaml_data_path=f'{self.current_file_path}/custom_dataset.yaml'
        )


    def model_test(self
                   , image_to_blurr: str = None
                   , project: str = 'detect'
                   , name: str = 'test'
                   , save: bool = True
                   , exist_ok: bool = True):
        
        if image_to_blurr is None:
            img_source = f'{self.current_file_path}/teste/DUH27ZB3JZL4FLRJXF4BMWD3XQ.jpg'
        else:
            img_source = image_to_blurr

        results = self.img_blurr.predict(
            image_to_blurr=img_source
            , save=save
            , project=project
            , name=name
            , exist_ok=exist_ok
        )        

        return results


    def blurr_plate(self
                   , input_image = None
                   , output_img_path = None):
        
        logging.info(f"input_image: {input_image}")
        logging.info(f"output_img_path: {output_img_path}")         
        if input_image is None:
            input_img = f'{self.current_file_path}/teste/DUH27ZB3JZL4FLRJXF4BMWD3XQ.jpg'
        else:
            input_img = input_image

        if output_img_path is None:
            output_image_path = f'{self.current_file_path}/blurred_img'
        else:
            output_image_path = output_img_path  

        os.makedirs(output_image_path, exist_ok=True)     

        logging.info(f"input_img: {input_img}")
        logging.info(f"output_image_path: {output_image_path}") 

        self.img_blurr.blurr_image(
            input_image = input_img
            , output_img_path= output_image_path
        )   


current_file = os.path.abspath(__file__)
current_file_path = os.path.dirname(current_file) 


#plate_blurr = PlateBlurr()
#plate_blurr.model_train()


plate_blurr = PlateBlurr(f'{current_file_path}/runs/detect/train/weights/best.pt')
##plate_blurr.model_test()
plate_blurr.blurr_plate()
