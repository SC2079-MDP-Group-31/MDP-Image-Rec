# capture_and_send.py
import io
import os
import json
import requests
from picamera import PiCamera
from time import sleep

# Your target server endpoint
UPLOAD_URL = "http://192.168.31.2:5000/image"

def snap_and_rec(url, obstacle_id_with_signal: str) -> None:
        """
        RPi snaps an image and calls the API for image-rec.
        The response is then forwarded back to the android
        :param obstacle_id_with_signal: the current obstacle ID followed by underscore followed by signal
        """
        obstacle_id, signal = obstacle_id_with_signal.split("_")
        self.logger.info(f"Capturing image for obstacle id: {obstacle_id}")
        self.android_queue.put(AndroidMessage(
            "info", f"Capturing image for obstacle id: {obstacle_id}"))
        url = f"http://{API_IP}:{API_PORT}/image"
        filename = f"{int(time.time())}_{obstacle_id}_{signal}.jpg"

        con_file = "PiLCConfig9.txt"
        Home_Files = []
        Home_Files.append(os.getlogin())
        config_file = "/home/" + Home_Files[0] + "/" + con_file

        self.logger.debug("Requesting from image API")

        response = self.capture_and_send(url)

        if response.status_code != 200:
            self.logger.error(
                "Something went wrong when requesting path from image-rec API. Please try again.")
            return
        
        results = json.loads(response.content)

        # release lock so that bot can continue moving
        self.movement_lock.release()
        try:
            self.retrylock.release()
        except:
            pass

        self.logger.info(f"results: {results}")
        self.logger.info(f"self.obstacles: {self.obstacles}")
        self.logger.info(
            f"Image recognition results: {results} ({SYMBOL_MAP.get(results['image_id'])})")

        if results['image_id'] == 'NA':
            self.failed_obstacles.append(
                self.obstacles[int(results['obstacle_id'])])
            self.logger.info(
                f"Added Obstacle {results['obstacle_id']} to failed obstacles.")
            self.logger.info(f"self.failed_obstacles: {self.failed_obstacles}")
        else:
            self.success_obstacles.append(
                self.obstacles[int(results['obstacle_id'])])
            self.logger.info(
                f"self.success_obstacles: {self.success_obstacles}")
        self.android_queue.put(AndroidMessage("image-rec", results))
    
    def capture_and_send(url):
        # Initialize PiCamera
        camera = PiCamera()

        try:
            # Optional: give camera some time to adjust
            sleep(2)

            # Capture image to an in-memory stream
            stream = io.BytesIO()
            camera.capture(stream, format="jpeg")
            stream.seek(0)

            # Send the image via POST
            files = {"file": ("image.jpg", stream, "image/jpeg")}
            response = requests.post(url, files=files)

            print("Status Code:", response.status_code)
            print("Response:", response.text)

            return response

        finally:
            camera.close()

if __name__ == "__main__":
    capture_and_send(UPLOAD_URL)
