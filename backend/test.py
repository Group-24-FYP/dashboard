from flask import Flask, jsonify

app = Flask(__name__)
@app.route('/api/data')
def get_data():
    data = {'uncertainty_score': '97', 'status': 'Defected Product' , 'ASM': '../../image/bottle.png'}
    return jsonify(data)

if __name__ == '__main__':
    app.run(use_reloader=True)