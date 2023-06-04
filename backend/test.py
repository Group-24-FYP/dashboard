from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/data')
def get_data():
    #image_path = request.json.get('image_path')
    data = {'uncertainty_score': '97', 'status': 'Defected Product'}
    #data = {'uncertainty_score': '97', 'status': 'Defected Product' , 'ASM': image_path}
    return jsonify(data)

if __name__ == '__main__':
    app.run(use_reloader=True)