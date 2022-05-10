from flask import Flask, jsonify, render_template, redirect, request
import joblib
import js2py
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return render_template("new.html")


@app.route("/predict", methods=["GET", "POST"])
def result():
    # item_weight = float(request.form['item_weight'])
    # item_visibility = float(request.form['item_visibility'])
    item_fat_content = float(request.form['item_fat_content'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

    X = np.array([[item_fat_content, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    print("----------------------------xxx-------------------------------")
    print(X)
    scaler_path = 'models/sc.sav'

    sc = joblib.load(scaler_path)

    X_std = sc.transform(X)

    model_path = 'models/rf.sav'

    model = joblib.load(model_path)

    Y_pred = model.predict(X_std)

    print("--------------------------")
    print(Y_pred)

    if item_fat_content == 0:
        item_fat_content = 'High Fat'
    elif item_fat_content == 1:
        item_fat_content = 'Low Fat'
    else:
        item_fat_content = 'Regular'

    if item_type == 0:
        item_type = 'Baking Goods'
    elif item_type == 1:
        item_type = 'Breads'
    elif item_type == 2:
        item_type = 'Breakfasts'
    elif item_type == 3:
        item_type = 'Canned'
    elif item_type == 4:
        item_type = 'Diary'
    elif item_type == 5:
        item_type = 'Frozen Foods'
    elif item_type == 6:
        item_type = 'Fruits and Vegetables'
    elif item_type == 7:
        item_type = 'Hard Drinks'
    elif item_type == 8:
        item_type = 'Health and Hygiene'
    elif item_type == 9:
        item_type = 'Household'
    elif item_type == 10:
        item_type = 'Meat'
    elif item_type == 11:
        item_type = 'Others'
    elif item_type == 12:
        item_type = 'Seafood'
    elif item_type == 13:
        item_type = 'Snack Foods'
    elif item_type == 14:
        item_type = 'Soft Drinks'
    else:
        item_type = 'Starchy Foods'

    if outlet_size == 0:
        outlet_size = 'High'
    elif outlet_size == 1:
        outlet_size = 'Medium'
    else:
        outlet_size = 'Small'

    if outlet_location_type == 0:
        outlet_location_type = 'Tier-1'
    elif outlet_location_type == 1:
        outlet_location_type = 'Tier-2'
    else:
        outlet_location_type = 'Tier-3'

    if outlet_type == 0:
        outlet_type = 'Grocery Store'
    elif outlet_type == 1:
        outlet_type = 'Supermarket Type1'
    elif outlet_type == 1:
        outlet_type = 'Supermarket Type2'
    else:
        outlet_type = 'Supermarket Type3'


    return render_template("prediction.html", result=int(Y_pred), itemFatContent=item_fat_content, itemType=item_type, itemMrp=item_mrp, outletEstablishmentYear=outlet_establishment_year, outletSize=outlet_size, outletLocationType=outlet_location_type, outletType=outlet_type )

@app.route('/second')
def second():
    return render_template('new.html')

# ---------------------------------------------------starting code ----------------------------------------------

'''
This route (end-point) simply returns all the required data like item fat content,
item tpe, etc. It is get request.
'''

@app.route("/api/required-data", methods=["GET"])
def getRequredData ():
    return jsonify(get_required_data())


'''
This route returns prediction data. This is post request. So that user send all the
requred data for the sales prediction.
'''

@app.route("/api/predict-sales", methods=["POST"])
def predict_sales():
    input_data = request.get_json(force=True) 
    verify_request= verify_request_data(input_data)
    
    if not verify_request["isVerified"]:
        return {"message": verify_request["message"]}, 400

    # Convert data into float number 

    item_fat_content = float(input_data['item_fat_content'])
    item_type = float(input_data['item_type'])
    item_mrp = float(input_data['item_mrp'])
    outlet_establishment_year = float(input_data['outlet_establishment_year'])
    outlet_size = float(input_data['outlet_size'])
    outlet_location_type = float(input_data['outlet_location_type'])
    outlet_type = float(input_data['outlet_type'])

    # Let's add all data into array
    
    X = np.array([[item_fat_content, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    # Let's load pre-trained modal

    scaler_path = 'models/sc.sav'
    sc = joblib.load(scaler_path)
    X_std = sc.transform(X)
    model_path = 'models/rf.sav'
    model = joblib.load(model_path)

    # Now, Predict the data

    sales_prediction = model.predict(X_std)
    
    # Now, response data with prediction value 

    required_data = get_required_data()
    item_fat_content_data = required_data["item_fat_content"]
    item_type_data = required_data["item_type"]
    outlet_size_data = required_data["outlet_size"]   
    outlet_type_data = required_data["outlet_type"]
    outlet_location_data = required_data["outlet_location"]

    item_fat_content_name = [x for x in item_fat_content_data if x[0] == int(item_fat_content)]
    item_type_name = [x for x in item_type_data if x[0] == int(item_type)]
    outlet_size_name = [x for x in outlet_size_data if x[0] == int(outlet_size)]
    outlet_type_name = [x for x in outlet_type_data if x[0] == int(outlet_type)]
    outlet_location_name = [x for x in outlet_location_data if x[0] == int(outlet_location_type)]

    return jsonify({
        "item_fat_content": item_fat_content_name[0] if len(item_fat_content_name) else [],
        "item_type": item_type_name[0] if len(item_type_name) else [],
        "outlet_size": outlet_size_name[0] if len(outlet_size_name) else [],
        "outlet_type": outlet_type_name[0] if len(outlet_type_name) else [],
        "outlet_location": outlet_location_name[0] if len(outlet_location_name) else [],
        "item_mrp": int(item_mrp),
        "outlet_establishment_year": int(outlet_establishment_year),
        "sales_prediction": sales_prediction[0]
    })


# Function that returns all the required data.

def get_required_data():
    item_fat_content = [[0, "High Fat"],[1, "low Fat"], [3, "Regular"]]
    item_type = [
            [0, "Baking Goods"], 
            [1, "Breads"], 
            [2, "Breakfasts"], 
            [3, "Canned"],
            [4, "Diary"],
            [5, "Frozen Foods"],
            [6, "Fruits and Vegetables"],
            [7, "Hard Drinks"],
            [8, "Health and Hygiene"],
            [9, "Household"],
            [10, "Meat"],
            [11, "Others"],
            [12, "Seafood"],
            [13, "Snack Foods"],
            [14, "Soft Drinks"],
            [15, "Starchy Foods"]
        ]
        
    outlet_size = [
        [0, "High"],
        [1, "Medium"],
        [2, "Small"]
    ]

    outlet_type = [
        [0, "Grocery Store"],
        [1, "Supermarket Type1"],
        [2, "Supermarket Type2"],
        [3, "Supermarket Type3"]
    ]

    outlet_location = [
        [1, "Tier 1"],
        [2, "Tier 2"],
        [3, "Tier 3"]
    ]

    return { 
        "item_fat_content":item_fat_content, 
        "item_type": item_type, 
        "outlet_size": outlet_size,
        "outlet_type": outlet_type,
        "outlet_location": outlet_location
    }


# Function to verify incoming data

def verify_request_data (data):
    try:
        if "item_fat_content" in data:
            if data["item_fat_content"] < 0:
                return {"isVerified": False, "message": "Invalid fat content"}
        else:
            return {"isVerified": False, "message": "Fat content is required"}

        if "item_type" in data:
            if data["item_type"] < 0:
                return {"isVerified": False, "message": "Invalid item type"}
        else:
            return {"isVerified": False, "message": "Item type is required"}

        if "item_mrp" in data:
            if data["item_mrp"] <= 0:
                return {"isVerified": False, "message": "Invalid MRP price"}
        else:
            return {"isVerified": False, "message": "Item's MRP price is required"}

        if "outlet_establishment_year" in data:
            if data["outlet_establishment_year"] < 1987:
                return {"isVerified": False, "message": "Invalid outlet establishment year"}
        else:
            return {"isVerified": False, "message": "Outlet establishment year is required"}

        if "outlet_size" in data:
            if data["outlet_size"] < 0:
                return {"isVerified": False, "message": "Invalid outlet size"}
        else:
            return {"isVerified": False, "message": "Outlet size is required"}

        if "outlet_location_type" in data:
            if data["outlet_location_type"] < 0:
                return {"isVerified": False, "message": "Invalid outlet location type"}
        else:
            return {"isVerified": False, "message": "Outlet location type is required"}

        if "outlet_type" in data:
            if not data["outlet_type"] > 0:
                return {"isVerified": False, "message": "Invalid item outlet type"}
        else:
            return {"isVerified": False, "message": "Outlet type is required"}

        return {"isVerified": True, "message": "Data is verified"}
    except:
        return {"isVerified": False, "message": "Invalid Data"}



if __name__ == "__main__":
    app.run(debug=True, port=9457)


