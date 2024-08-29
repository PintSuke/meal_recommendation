import argparse
import ast
import math
import os
import pickle
import random
import warnings
from ast import literal_eval
from collections import OrderedDict
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.inception import InceptionOutputs

from models import *

from googleapiclient.discovery import build

import pdb
from tqdm import tqdm

class Nutrition_RGBD_Inference(Dataset):
    def __init__(self, RGB_image, depth_image, transform=None):
        self.images = []
        self.images_rgbd = []
        for image in RGB_image:
            self.images += [image]
        for d_image in depth_image:
            self.images_rgbd += [d_image]
        self.transform = transform


    def __getitem__(self, index):
        try:
            img_rgb = self.images[index]
            img_rgbd = self.images_rgbd[index]
        except Exception as e:
            print(f"Error loading image at index {self.images[index]}: {e}")
            return None, None
        
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd.convert('RGB'))

        return img_rgb, img_rgbd

    def __len__(self):
        return len(self.images)
    
def number_to_rgb_link(number):
    return'https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/dish_'+str(number)+'/rgb.png'

def get_InferenceDataLoader(RGB_image, depth_image,):
    inference_transform = transforms.Compose([
                            transforms.Resize((320, 448)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    inferenceset = Nutrition_RGBD_Inference(RGB_image, depth_image, transform=inference_transform)
    inference_loader = DataLoader(inferenceset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)

    return inference_loader

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def inference(inference_loader, net, net2, net_cat):
    predicted_value = [[],[],[],[],[]]
    net.eval()
    net2.eval()
    net_cat.eval()

    epoch_iterator = tqdm(inference_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, x in enumerate(epoch_iterator): # testloader
            inputs_rgb = x[0].to(device)
            inputs_rgbd = x[1].to(device)

            outputs_rgb = net(inputs_rgb)
            p2, p3, p4, p5 = outputs_rgb

            outputs_rgbd = net2(inputs_rgbd)
            d2, d3, d4, d5 = outputs_rgbd

            outputs = net_cat([p2, p3, p4, p5], [d2, d3, d4, d5])

            predicted_value[0].append(outputs[0])
            predicted_value[1].append(outputs[1])
            predicted_value[2].append(outputs[2])
            predicted_value[3].append(outputs[3])
            predicted_value[4].append(outputs[4])
    return predicted_value


def fetch_image(numbers):
    img_rgb = []
    img_rgbd = []
    for number in numbers:
        RGB_url = number_to_rgb_link(number)
        depth_url = RGB_url.replace('rgb.png', 'depth_raw.png')

        RGB_response = requests.get(RGB_url)
        depth_response = requests.get(depth_url)

        img_rgb.append(Image.open(BytesIO(RGB_response.content)))
        img_rgbd.append(Image.open(BytesIO(depth_response.content)))
    return img_rgb, img_rgbd

def predict_module1(image_rgb = [], image_rgbd = [], from_image = False, image_numbers = []):
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # print("Loading models")
    net = torch.load('net.pkl')
    net2 = torch.load('net2.pkl')
    net_cat = torch.load('net_cat.pkl')

    # print("Default setup")
    # args = parser_setup(parser)
    if from_image:
        RGB_image = image_rgb
        depth_image = image_rgbd
    else:
        if image_numbers != []:
            RGB_image, depth_image = fetch_image(image_numbers)
        else:
            return [[0],[0],[0],[0],[0]]
    
    # print("gonna create loader")
    inference_loader = get_InferenceDataLoader(RGB_image, depth_image,)
    # print("Inferencing")
    inference_result = inference(inference_loader, net, net2, net_cat)
    # print("finish inference result", inference_result)
    return inference_result


def update_ratings(dataframe, original_ratings, dynamic_flag, i):
    if dynamic_flag:
        for index, row in dataframe.iterrows():
            dataframe.loc[index, 'rating'] += (original_ratings.loc[index, 'rating'] - dataframe.loc[index, 'rating']) / (7 - i + 1)

def decrement_ratings(dataframe, original_ratings, dynamic_flag, i):
    if dynamic_flag:
        for index, row in dataframe.iterrows():
            dataframe.loc[index, 'rating'] -= (original_ratings.loc[index, 'rating'] - dataframe.loc[index, 'rating']) / (7 - i + 1)

def get_meal_info(meal_sel_info, recipes, i):
    meal_order= {
        'a': "Breakfast",
        'b': "Lunch",
        'c': "Snack",
        'd': "Dinner"
        }
    print(f"The recommended recipe for {meal_order[meal_sel_info]}, named \"{recipes['name'].iloc[i]}\", needs {recipes['minutes'].iloc[i]} minutes of cooking time, and needs also "
          f"{recipes['n_ingredients'].iloc[i]} ingredients. The recipe has {recipes['calories'].iloc[i]} calories, "
          f"{recipes['total fat (%)'].iloc[i]} total fat (%), {recipes['sugar (%)'].iloc[i]} sugar (%), "
          f"{recipes['sodium (%)'].iloc[i]} sodium (%), {recipes['protein (%)'].iloc[i]} protein (%), "
          f"{recipes['carbohydrates (%)'].iloc[i]} carbohydrates (%), {recipes['saturated fat (%)'].iloc[i]} saturated fat (%).\n")

def meal_plan_loop(user_option, meal_data, i):
    while user_option == 'c':
        meal_sel_info = input("About which meal do you want us to provide you with information (a, b, c, d)? :\n a)Breakfast \n b)Lunch \n c)Snack \n d)Dinner \n")
        get_meal_info(meal_sel_info, meal_data[meal_sel_info], i)

        next_option = input("What do you want to do next (a, b, c) :\n a)Proceed to the next day \n b)I want you to change the menu for a meal \n c)I want more information about a meal \n")
        if next_option == 'a':
            user_option = 'a'
        elif next_option == 'b':
            user_option = 'b'
        

def process_meal_selection(meal, action_type, meal_data, user_id):
    """Process the meal based on user input to either exclude or include in future recommendations."""
    count_key = f"count_{action_type[:2]}_{meal[:2]}"  # Simplified indexing for counts
    meal_data[count_key] += 1
    ingredients_set = set(meal_data[f"{meal}_ingredients"])
    new_ingredients = ast.literal_eval(meal_data['recipes'][meal].head(1)['ingredients'].values[0])
    
    if meal_data[count_key] == 1:
        ingredients_set.update(new_ingredients)
    else:
        common_ingredients = ingredients_set.intersection(new_ingredients)
        for index, row in meal_data['recipes'][meal].iterrows():
            recipe_ingredients = set(eval(row['ingredients']))
            if recipe_ingredients.intersection(common_ingredients):
                adjust_recipe_rating(row, index, meal_data, user_id)
    
    meal_data['recipes'][meal] = meal_data['recipes'][meal].drop(meal_data['recipes'][meal].head(1).index)

def adjust_recipe_rating(row, index, meal_data, user_id):
    """Adjust recipe ratings based on common ingredients."""
    common_len = len(set(eval(row['ingredients'])).intersection(meal_data['common_ingredients']))
    weight = common_len * round(Mean.loc[Mean['user_id'] == user_id, 'rating'].iloc[0] % 0.1, 3)
    meal_data['recipes'][row['meal']].loc[index, 'rating'] -= weight
    meal_data['recipes'][row['meal']] = meal_data['recipes'][row['meal']].sort_values(by='rating', ascending=False)

def get_user_input(prompt):
    """Get validated user input."""
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ['a', 'b', 'c', 'd']:  # Add specific checks as needed
            return user_input
        else:
            print("Invalid input, please choose a valid option.")

def format_steps(unformat_text):
    steps = unformat_text.strip("[]").split("', '")

    # Clean each step by stripping extra quotes
    steps = [step.strip("'") for step in steps]

    # Now format it as a list with each step on a new line
    formatted_steps = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(steps))
    
    return formatted_steps

def pick_a_recipe(arr_day, meal_to_change, recipe_list, upper_num, lower_num, user, keyword = 'calories'):
    recipes_filtering = recipe_list
    upper_condition = recipes_filtering[keyword].astype(float)<=upper_num
    lower_condition = recipes_filtering[keyword].astype(float)>=lower_num

    
    if not recipes_filtering[upper_condition][lower_condition].empty:   
        recipes_filtering=recipes_filtering[upper_condition][lower_condition]
        flag_dynamic[meal_to_change][1] = True
        weight=round(Mean.loc[Mean['user_id'] == user, 'rating'].iloc[0] % 0.1, 4)
        
        recipes_filtering.loc[upper_condition, 'rating'] +=weight
        recipes_filtering.loc[lower_condition, 'rating'] +=weight
        arr_day[meal_to_change, 0] = recipes_filtering['name'].iloc[0]  # Recipe name
        arr_day[meal_to_change, 1] = format_steps(recipes_filtering['steps'].iloc[0])  # Recipe steps
        arr_day[meal_to_change, 2] = recipes_filtering['calories'].iloc[0]
        arr_day[meal_to_change, 3] = recipes_filtering['total fat (%)'].iloc[0]
        arr_day[meal_to_change, 4] = recipes_filtering['sugar (%)'].iloc[0]
        arr_day[meal_to_change, 5] = recipes_filtering['sodium (%)'].iloc[0]
        arr_day[meal_to_change, 6] = recipes_filtering['protein (%)'].iloc[0]
        arr_day[meal_to_change, 7] = recipes_filtering['saturated fat (%)'].iloc[0]
        arr_day[meal_to_change, 8] = recipes_filtering['carbohydrates (%)'].iloc[0]
        arr_day[meal_to_change, 9] = recipes_filtering['ingredients'].iloc[0]
    
    else:
        arr_day[meal_to_change, 0] = "Sorry, We don't have recipe suit for this meal"  # Recipe name
        arr_day[meal_to_change, 1] = "-"  # Recipe steps
        arr_day[meal_to_change, 2] = "0"
        arr_day[meal_to_change, 3] = "0"
        arr_day[meal_to_change, 4] = "0"
        arr_day[meal_to_change, 5] = "0"
        arr_day[meal_to_change, 6] = "0"
        arr_day[meal_to_change, 7] = "0"
        arr_day[meal_to_change, 8] = "0"
        arr_day[meal_to_change, 9] = "None"
    return arr_day[meal_to_change, 0] ,arr_day[meal_to_change, 1], arr_day[meal_to_change, 2], arr_day[meal_to_change, 3], arr_day[meal_to_change, 4], arr_day[meal_to_change, 5], arr_day[meal_to_change, 6], arr_day[meal_to_change, 7], arr_day[meal_to_change, 8], arr_day[meal_to_change, 9]

def meal_recommendation(user = 37449, meal_checklist = [True, True, True, True], image_rgb = [], image_rgbd = [], from_image = False, image_numbers = []):
    global Mean
    global flag_dynamic

    similarity = pd.read_pickle("similarity_matrix.pkl")
    k=math.trunc(math.sqrt(len(similarity.index))) #I will apply K-nearest neighbors algorithm so I calculate the k finding the square root of the number of samples in above dataset
    knn=similarity.apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:k+1].index), axis=1) #create a dataframe with the k neighbors of each user
    knn.drop(columns=knn.columns[0], 
            axis=1, 
            inplace=True)
    with open('prepared_variables.pkl', 'rb') as f:
        loaded_variables = pickle.load(f)

    # Accessing the loaded variables
    rating_avg = loaded_variables['rating_avg']
    recipes_v1 = loaded_variables['recipes_v1']
    pivot_table = loaded_variables['pivot_table']
    Mean = loaded_variables['Mean']
    # Default user
    # user = 37449

    #find all K-nearest neighbors recipes
    all_users_recipes=rating_avg.astype({"recipe_id":str}).groupby(by = 'user_id')['recipe_id'].apply(lambda x:','.join(x)) #the rated recipes of each unique user in dataset
    all_k_nearest_neighbours_of_user=knn[knn.index==user].values.squeeze().tolist() # the k nearest neighbours(ids) of the user we want to recommend to
    recipes_of_k_nearest_neighbours=all_users_recipes[all_users_recipes.index.isin(all_k_nearest_neighbours_of_user)] # all the rated recipes of the k nearest neighbours(with neighbors ids) of the user we want to recommend to
    neighbours_recipes=','.join(recipes_of_k_nearest_neighbours.values).split(',') #we keep only the neighbours recipes ids

    #find the recipes already rated by the user to exclude from recipes to recommend
    user_recipes=pivot_table.columns[pivot_table[pivot_table.index==user].notna().any()].tolist() #In the pivot table with NaNs we find all the actual ratings of the user we want to recommend to

    #find all the possible recipes for recommendation after the exclusion

    possible_recipes=[int(item) for item in neighbours_recipes if item not in user_recipes]
    #list(map(int,list(set(neighbours_recipes)-set(list(map(str, user_recipes)))))) #is a list of integers(recipes ids)
    recipes_predictions=[]
    user_avg=Mean.loc[Mean['user_id']==user,'rating'].values[0] #the avarage rating of the user
    sum1=0
    sum2=0
    for j in possible_recipes:
        for i in all_k_nearest_neighbours_of_user:
            
            sim_with_user=similarity.loc[user,i] #the similarity of neighbour with the user we want to recommend to
            rating_i_to_j=poss_recipe_col=pivot_table.loc[i, j] if not pd.isna(pivot_table.loc[i, j]) else 0 # neighbour's rating of possible recipe
            sum1=sum1+sim_with_user*rating_i_to_j #the sum over neighbours (Rating*Similarity)
            sum2=sum2+sim_with_user #the sum over neighbours(similarity)
        normalized_pred=sum1/sum2 # the predicted normalized rating of the recipe
        actual_predicted_rating=user_avg+normalized_pred #the actual predicted rating which is calculated with avarage user rating plus the normalized predicted rating
        recipes_predictions.append((actual_predicted_rating))
    #find the recommendations with their id and after using the ids we find also their names
    rec_df=pd.DataFrame({'recipe_id':possible_recipes,'rating':recipes_predictions})
    top_recommendations=rec_df.sort_values(by='rating',ascending=False)

    top_recommendations.rename(columns = {'recipe_id':'id'}, inplace = True)
    recipes_final=top_recommendations.merge(recipes_v1,how='inner',on='id')
    recipe_names=recipes_final.name.values.tolist()
    recipes_final=recipes_final.sort_values(by='rating',ascending=False).reset_index(drop=True)
    recipes_final_breakfast = recipes_final[recipes_final['breakfast'] == True]
    recipes_final_lunch = recipes_final[recipes_final['lunch'] == True]
    recipes_final_snack = recipes_final[recipes_final['snacks'] == True]
    recipes_final_dinner = recipes_final[recipes_final['dinner'] == True]
    recipes_final_list = [recipes_final_breakfast,recipes_final_lunch,recipes_final_snack,recipes_final_dinner]

    or_rat = []
    for i, meal_list in enumerate(recipes_final_list):
        or_rat.append(meal_list.copy())
    
    flag_dynamic = [[False, False]]*4 # [[breakfast, breakfast_filter],[lunch, lunch_filter],[snack, snack_filter],[dinner, dinner_filter]]

    i = 0

    arr_day = np.empty((4,10), dtype=object) # Breakfast name, step, calories -> Lunch -> Snack
    cal_daily_need = 2000
    # user_default_option = 37449
    # breakfast_checkbox, lunch_checkbox, snack_checkbox, dinner_checkbox = True, True, True, True # True =  did not eat, change to False if eaten (checkbox chcked)
    # breakfast_checkbox = False # When checked
    snack_checkbox = meal_checklist[2] # API
    meal_havent_eat = meal_checklist # Meal that we have to generate recipe for [breakfast_checkbox, lunch_checkbox, snack_checkbox, dinner_checkbox]
    total_unit_to_generate = sum(meal_havent_eat)*2 - snack_checkbox # 2 Unit for All except snack is 1

    # outputs_temp = [
    #     [200, 150, 300],  # calories for different items
    #     [50, 75, 60],     # mass for different items
    #     [10, 8, 12],      # fat for different items
    #     [20, 30, 40],     # carbohydrates for different items
    #     [5, 7, 6]         # protein for different items
    # ]

    i = 0
    module1_output = predict_module1(image_rgb, image_rgbd, from_image, image_numbers)
    calories = module1_output[0][i]
    mass =  module1_output[1][i]
    fat = module1_output[2][i]
    carb = module1_output[3][i]
    protein = module1_output[4][i]

    calories_eaten = float(sum(module1_output[0]))
    cal_daily_need -= calories_eaten
    cal_per_unit = 0 if total_unit_to_generate == 0 else cal_daily_need/total_unit_to_generate
    cal_need_each_meal = [meal_bool * 2 * cal_per_unit for meal_bool in meal_havent_eat] 
    cal_need_each_meal[2] -= cal_per_unit*snack_checkbox

    for n, recipe_list in enumerate(recipes_final_list):
        if meal_havent_eat[n]:
            # print(arr_day, n,  cal_need_each_meal[n]+50, cal_need_each_meal[n]-50, 'calories')
            arr_day[n, 0], arr_day[n, 1], arr_day[n, 2], arr_day[n, 3], arr_day[n, 4], arr_day[n, 5], arr_day[n, 6], arr_day[n, 7], arr_day[n, 8], arr_day[n, 9] = pick_a_recipe(arr_day, n, recipe_list, cal_need_each_meal[n]+50, cal_need_each_meal[n]-50, user, 'calories')
    return arr_day

# Example of output
# x = meal_generation()
# print(x)
# [['egg white french toast'
#   'Step 1: in medium bowl beat egg whites well\nStep 2: add milk , vanilla and cinnamon to egg whites and continue to beat\nStep 3: dip bread slices into batter and coat on both sides\nStep 4: cook one of the following ways: place dipped bread in greased and heated skillet\nStep 5: cook by turning a few times to brown\nStep 6: or , place dipped bread on greased or oiled baking sheet\nStep 7: then broil in the oven , turning only once'
#   '182.4']
#  ['cream of potato soup'
#   'Step 1: in large saucepan , cook onions in butter until soft\nStep 2: add potatoes , chicken broth , parsley , thyme , celery seed , salt , and pepper to saucepan\nStep 3: simmer for 15 minutes\nStep 4: add milk to saucepan\nStep 5: puree half of soup and flour in blender\nStep 6: return to saucepan\nStep 7: heat through'      
#   '153.7']
#  ['popcorn  stove top'
#   'Step 1: put the oil in a 4 quart heavy pot and let it heat over medium heat for 30 seconds\nStep 2: stir in the kernels , turning with a spoon so that they are evenly covered with oil , then spread them in one layer on the bottom of the pot\nStep 3: cover the pot , leaving a small space at the edge for escaping steam\nStep 4: as soon as the first kernel pops , move the pot gently and continuously back and forth over medium-high heat until the popping stops\nStep 5: turn into a warm bowl\nStep 6: toss with melted butter , if desired , and salt to taste'
#   '90.2']
#  ['malibu   cranberry cocktail'
#   'Step 1: fill glass with ice\nStep 2: pour cranberry juice and malibu into glass\nStep 3: stir with straw or stirrer\nStep 4: sip & enjoy'
#   '162.8']]

# Function to search for images using Google Custom Search API
def search_image(query):
    # Your Google Custom Search API key and CX (Custom Search Engine ID)
    api_key = 'AIzaSyBvnsXb5Xqw-b0f1jgmSuENh_HSgBF7PPQ'
    cx = '45192a5a8ed354830'
    
    # Build the service object for the Google Custom Search API
    service = build("customsearch", "v1", developerKey=api_key)
    
    # Execute the search
    result = service.cse().list(q=query, cx=cx, searchType='image', num=1).execute()
    
    # Get the first image's link
    if 'items' in result:
        image_url = result['items'][0]['link']
        # website_url = result['items'][0]['link'] # full link
        website_url = result['items'][0]['displayLink'] # shorten link
        return image_url, website_url
    else:
        return None
    
def load_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Search and load image function from query
def snl_image(query):
    image_url = search_image(query)
    image = load_image(image_url)
    return image