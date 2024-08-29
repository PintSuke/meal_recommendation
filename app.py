from flask import Flask, json
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from mealplanapi import meal_recommendation, search_image
from PIL import Image
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = './upload_image'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Allowed extensions for files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
def get_upload_image(filename):
    if filename not in request.files:
        print(f'No {filename} part')
        return None, False
    file = request.files[filename]
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image = Image.open(file.stream)
        return image, True
    return None, False

@app.route('/', methods=['GET', 'POST'])
def index():
    # MEALS = {0:"Breakfast", 1:"Lunch", 2:"Snack", 3:"Dinner"}
    Meals = [True]*4
    sum_result = "Result will show here"
    recipes = []
    if request.method == 'POST':
        print("POST request click")
    
        user_type = int(request.form.get('userType'))
        print("user_type",user_type)

        # Retrieve the selectedImages data from the form
        selected_images = request.form.get('selectedImages')
        
        # Convert the JSON string to a Python list
        selected_images = json.loads(selected_images) if selected_images else []

        # Now `selected_images` is a list of selected image URLs
        # You can process it as needed, for example:
        print("Selected images:", selected_images)
        
        rgb_files = []
        depth_files = []
        have_rgb = False
        have_depth = False
        # Iterate through the uploaded files
        for key, file in request.files.items():
            if file and file.filename:  # Ensure the file is not empty
                # Check if the file is an RGB file
                if key.startswith("rgbFile"):
                    image, image_bool = get_upload_image(key)
                    rgb_files.append(image)
                    have_rgb = bool(have_rgb + image_bool)
                # Check if the file is a Depth file
                elif key.startswith("depthFile"):
                    image, image_bool = get_upload_image(key)
                    depth_files.append(image)
                    have_depth = bool(have_depth + image_bool)
    
        request_meals = request.form.getlist('meals')
        for meal in request_meals:
            Meals[int(meal)] = False

        print("generating")
        sum_result = meal_recommendation(user_type, Meals, rgb_files, depth_files, from_image = bool(have_rgb*have_depth), image_numbers = selected_images)
        
        # print(sum_result)
        # [[None None None None None None None None None None]
        # ['whatever floats your boat  brownies'
        # 'Step 1: preheat oven to 350f\nStep 2: grease an 8 inch square pan or line with foil\nStep 3: in a medium bowl combine melted butter and cocoa and stir until cocoa is dissolved\nStep 4: add sugar and mix well\nStep 5: add eggs one at a time and stir until well combined\nStep 6: stir in vanilla , flour and salt until you no longer see any flour\nStep 7: fold in "whatever floats your boat" !\nStep 8: spread in pan and bake for approximately 25 minutes\nStep 9: do not over-bake -- your brownies will come out dry\nStep 10: adjust time / temp accordingly for your oven\nStep 11: if you do the knife / toothpick test , it should come out with moist crumbs , not clean\nStep 12: cool completely before cutting into squares\nStep 13: for vegetarian omit the marshmallows\nStep 14: for double recipe , bake in 9x12 pan and add 5 minutes to baking time'
        # '390.7' ' 30.0' ' 161.0' ' 7.0' ' 12.0' ' 50.0' ' 17.0'
        # '[\'butter\', \'unsweetened cocoa\', \'sugar\', \'eggs\', \'vanilla\', \'flour\', \'salt\', \'chocolate chips\', \'raisins\', \'maraschino cherry\', \'nuts\', "m&m\'", "reese\'s pieces", \'miniature marshmallow\']'],
        # ['mama s creamed peas for sick tummies'
        # 'Step 1: melt butter in a large sauce pan\nStep 2: whisk in flour and allow to cook for 1 minute\nStep 3: slowly add milk , whisking the whole time to prevent lumps\nStep 4: add salt and pepper\nStep 5: allow to cook until sauce starts to thicken\nStep 6: add peas , stir and cook until peas are heated through about 5-7 minutes\nStep 7: can be served alone or over toast'
        # '238.7' ' 12.0' ' 38.0' ' 9.0' ' 22.0' ' 25.0' ' 10.0'
        # "['peas', 'milk', 'flour', 'butter', 'salt', 'pepper']"],
        # ['tapas   scallops in saffron   rioja wine sauce'
        # 'Step 1: in a saute pan over medium high heat , add olive oil and butter\nStep 2: when butter melts , add the shallots , saute for 30 seconds\nStep 3: add the scallops and wine\nStep 4: turn scallops and add saffron\nStep 5: season with salt\nStep 6: cook until scallops are golden brown , about 30 seconds to 1 minute\nStep 7: remove from heat and serve hot'
        # '458.3' ' 68.0' ' 0.0' ' 13.0' ' 19.0' ' 40.0' ' 1.0'
        # "['olive oil', 'shallot', 'butter', 'saffron', 'rioja wine', 'scallops', 'salt and pepper']"]]
        
        print("Finish generation")

        current_date = datetime.now().strftime("%B %d, %Y")  # Get current date in format like "August 19, 2024"
        recipes = []
        for item in sum_result:  # Skipping the first [None None None] entry
            if item[0] is not None:
                food_image, website_url = search_image(item[0])
                citation = f'Retrieved from <a href="http://{website_url}" target="_blank">{website_url}</a>, accessed on {current_date}.'
                ingredients_list = ""
                ingredients = item[9].strip("[]").split("', '")
                ingredients = [ingredient.strip("'") for ingredient in ingredients]
                for ingredient in ingredients:
                    ingredients_list += ingredient + ", "
                ingredients_list.strip(', ')
                recipe = {
                    'title': item[0],
                    'steps': item[1].split('\n'),
                    'ingredients': ingredients_list,
                    'Nutritional_values': 'calories: '+item[2] + 'kcal. total fat: '+item[3] + 'g. sugar: '+item[4] + 'g. sodium: '+item[5] + 'g. protein: '+item[6] + 'g. saturated fat:'+item[7] + 'g. carbohydrates:'+item[8]+'g.',
                    'image': food_image,
                    'citation': citation  # Add the formatted citation here
                }
                recipes.append(recipe)
        print("Meals", Meals)
        print("rgb_file", type(rgb_files), len(rgb_files), rgb_files)
        print("depth_file", type(depth_files), len(depth_files), depth_files, )
        return render_template('output.html', recipes=recipes)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)