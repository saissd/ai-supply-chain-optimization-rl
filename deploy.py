from flask import Flask, request, jsonify
from stable_baselines3 import DQN

app = Flask(__name__)

# ✅ Load trained RL model
model = DQN.load("models/supply_chain_dqn")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    inventory_level = data["inventory_level"]

    # ✅ Convert input to correct format
    obs = [[inventory_level]]
    action, _ = model.predict(obs)
    
    return jsonify({"action": int(action)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
