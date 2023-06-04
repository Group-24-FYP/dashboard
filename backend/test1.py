from flask import Flask, jsonify, request 
import load_and_evaluate_patchcore
import subprocess
import time
app = Flask(__name__)



image_path = '/home/mayooran/niraj/inference/patchcore/test_image/cap_001.png'


def run_patchcore_script(object_class):
    if object_class == 'capsule':
        model_path = "/home/mayooran/niraj/patchcore_main/patchcore-inspection/results/MVTecAD_Results_fewshot/capsule/models/mvtec_capsule"

    st = time.time()
    command = f"python load_and_evaluate_patchcore.py segmentation_map patch_core_loader \
                -p {model_path} \
                dataset --subdatasets {object_class} mvtec {image_path}"
    subprocess.run(command, shell=True)
    et = time.time()
    print("time", et-st)

run_patchcore_script(object_class='capsule')

"""
@app.route('/api/data')
def get_data():
    
    #frames = request.files.getlist('frames')
    #print(frames)
    run_patchcore_script(object_class='capsule')
    b_image_path = "/home/mayooran/mayooran/dashboard/dashboard/backend/segmentation_map/_seg.png"

    data = {'uncertainty_score': '97', 'status': 'Defected Product' , 'ASM': '../../image/bottle.png'}
    return jsonify(data)

if __name__ == '__main__':
    app.run(use_reloader=True)

"""
 