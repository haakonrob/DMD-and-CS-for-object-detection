from cv2 import waitKey
from .utils import isDigit

class cvInputParser:
    def handle_inputs(self, current_params):
        new_model = None
        params = None
        will_quit = False

        try:
            key = waitKey(1)

            if isDigit(key):
                i = int(chr(key))
                new_model = i
            elif key == ord('q'):
                will_quit = True 
            elif key == ord('w'):
                params = current_params
                params["max_rank"] += 1
            elif key == ord('a'):
                params = current_params
                params["N"] -= 5
            elif key == ord('s'):
                params = current_params
                params["max_rank"] -= 1
            elif key == ord('d'):
                params = current_params
                params["N"] += 5
            elif key == ord('-'):
                params = current_params
                params['downsample'] += 1
            elif key == ord('='):
                params = current_params
                params['downsample'] = max(1,params['downsample']-1)
            elif key == ord('t'):
                params = current_params
                params['T'] = 1/30 if params['T'] == 0 else 0
            elif key > -1: 
                print(key)
        except Exception as e:
            print(e)

        return new_model, params, will_quit