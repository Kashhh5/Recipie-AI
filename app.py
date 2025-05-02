import streamlit as st
import pandas as pd
import random
import time
import os
from collections import Counter
from fuzzywuzzy import process
import speech_recognition as sr
from PIL import Image, ImageOps
from streamlit_option_menu import option_menu
import base64

# ========== Configuration ==========
st.set_page_config(
    page_title="Recipe Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Background Setup ==========
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 10px;
            margin: 2rem 0;
        }}
        .recipe-card {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .recipe-card:hover {{
            transform: translateY(-5px);
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background("new.jpg")

# ========== Data Loading ==========
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("Food_Recipes_With_Cuisine.csv")
        df.columns = df.columns.str.strip().str.lower()
        
        # ====== Add Diet Classification Here ======
        # ====== Add Diet Classification Here ======
        # In the load_data() function, update the diet classification:
        if 'diet' not in df.columns:
            # Create empty diet column first
            df['diet'] = 'Nonvegetarian'  # default
            
            # Define comprehensive lists of non-veg ingredients (now including eggs)
            non_veg_ingredients = [
                'chicken', 'meat', 'beef', 'pork', 'fish', 'mutton', 'lamb', 
                'seafood', 'egg', 'eggs', 'bacon', 'sausage', 'ham', 'shrimp',
                'prawn', 'crab', 'turkey', 'duck', 'goose', 'venison', 'animal'
            ]
            
            dairy_ingredients = [
                'milk', 'cheese', 'yogurt', 'butter', 'cream', 'ghee', 'curd',
                'paneer', 'dairy', 'whey', 'casein'
            ]
            
            # Mark vegetarian (no meat/eggs but may contain dairy)
            veg_mask = ~df['cleaned_ingredients'].str.lower().str.contains(
                '|'.join(non_veg_ingredients), regex=True
            )
            df.loc[veg_mask, 'diet'] = 'Vegetarian'
            
            # Mark vegan (no animal products at all - no meat, eggs, or dairy)
            vegan_mask = veg_mask & ~df['cleaned_ingredients'].str.lower().str.contains(
                '|'.join(dairy_ingredients + ['honey']), regex=True
            )
            df.loc[vegan_mask, 'diet'] = 'Vegan'
        # ====== End of Diet Classification ======
        
        if 'image' in df.columns:
            df['image_path'] = df['image'].apply(
                lambda x: f"images/{x}" if pd.notnull(x) and os.path.exists(f"images/{x}") else None
            )
        
        if 'unnamed: 0' in df.columns:
            df.drop(columns=["unnamed: 0"], inplace=True)
            
        if 'title' in df.columns:
            df.drop_duplicates(subset="title", inplace=True)
        else:
            st.error("⚠️ 'title' column is missing!")
            
        return df
    
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# ========== Helper Functions ==========
def filter_recipe(cuisine_type, diet_type, meal_type, taste_profile, ingredients_input=""):
    filtered_df = df.copy()
    
    # --- 1. Standardize diet_type (fix typos like "Non-Vegetarain") ---
    diet_type = diet_type.lower().replace("-", "").replace(" ", "")
    if "nonveg" in diet_type or "nonvegetarian" in diet_type:
        diet_type = "nonvegetarian"
    elif "veg" in diet_type and "vegan" not in diet_type:
        diet_type = "vegetarian"
    
    # --- 2. Cuisine Filter ---
    if cuisine_type != "Any" and "cuisine" in df.columns:
        filtered_df = filtered_df[filtered_df["cuisine"].str.lower() == cuisine_type.lower()]
    
    # --- 3. Diet Filter (BRUTE-FORCE ENFORCEMENT) ---
    if diet_type != "any":
        # VEGETARIAN: No meat/fish/eggs
        if diet_type == "vegetarian":
            filtered_df = filtered_df[
                ~filtered_df["cleaned_ingredients"].str.contains(
                    r'(chicken|meat|beef|pork|fish|mutton|lamb|seafood|egg|shrimp)s?',
                    case=False, regex=True, na=False
                )
            ]
        
        # VEGAN: No animal products at all
        elif diet_type == "vegan":
            filtered_df = filtered_df[
                ~filtered_df["cleaned_ingredients"].str.contains(
                    r'(meat|fish|egg|milk|cheese|yogurt|butter|cream|ghee|paneer|honey)',
                    case=False, regex=True, na=False
                )
            ]
        
        # NON-VEGETARIAN: MUST contain meat/fish/eggs, NO veg ingredients
        elif diet_type == "nonvegetarian":
            # Step 1: Must have meat/fish/eggs
            filtered_df = filtered_df[
                filtered_df["cleaned_ingredients"].str.contains(
                    r'(chicken|meat|beef|pork|fish|mutton|lamb|seafood|egg)s?',
                    case=False, regex=True, na=False
                )
            ]
            # Step 2: Remove ANY veg contamination (even in title)
            filtered_df = filtered_df[
                ~filtered_df["title"].str.contains(
                    r'(paneer|tofu|vegetarian|vegan)',
                    case=False, regex=True, na=False
                )
            ]
            # Step 3: Purge veg ingredients (even if user searches for them)
            filtered_df = filtered_df[
                ~filtered_df["cleaned_ingredients"].str.contains(
                    r'(paneer|tofu|soy)',
                    case=False, regex=True, na=False
                )
            ]
    
    # --- 4. Other Filters (Meal Type, Taste) ---
    if meal_type != "Any" and "meal_type" in df.columns:
        filtered_df = filtered_df[filtered_df["meal_type"].str.lower() == meal_type.lower()]
    if taste_profile != "Any" and "taste_profile" in df.columns:
        filtered_df = filtered_df[filtered_df["taste_profile"].str.lower() == taste_profile.lower()]
    
    # --- 5. Ingredients Filter (Diet-Aware) ---
    if ingredients_input:
        ingredients = [ing.strip().lower() for ing in ingredients_input.split(",") if ing.strip()]
        # NON-VEG MODE: IGNORE vegetarian ingredients (even if searched)
        if diet_type == "nonvegetarian":
            ingredients = [ing for ing in ingredients if ing not in ["paneer", "tofu", "soy"]]
            if not ingredients:  # Only veg ingredients were entered
                return pd.DataFrame()  # Return empty
        
        for ing in ingredients:
            filtered_df = filtered_df[
                filtered_df["cleaned_ingredients"].str.contains(
                    ing, case=False, regex=False, na=False
                )
            ]
    
    return filtered_df

def smart_ingredient_suggestions(user_input, all_ingredients_list, diet_type="Any"):
    if not user_input:
        return []
    
    # Filter out non-veg ingredients if vegetarian/vegan is selected
    if diet_type.lower() in ["vegetarian", "vegan"]:
        non_veg_ings = ["chicken", "meat", "beef", "pork", "fish", "mutton", 
                       "lamb", "seafood", "egg", "bacon", "sausage"]
        all_ingredients_list = [
            ing for ing in all_ingredients_list 
            if not any(non_veg in ing.lower() for non_veg in non_veg_ings)
        ]
    
    matches = process.extract(user_input.lower(), all_ingredients_list, limit=5)
    return [match[0] for match in matches if match[1] > 70]

def fetch_nutritional_info(recipe_title):
    return {
        "Calories": f"{random.randint(200, 800)} kcal",
        "Protein": f"{random.randint(5, 40)}g",
        "Carbs": f"{random.randint(20, 100)}g",
        "Fats": f"{random.randint(5, 35)}g",
        "Fiber": f"{random.randint(2, 15)}g"
    }

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        audio = recognizer.listen(source, timeout=5)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.error("Sorry, could not understand the audio.")
        return ""
    except Exception as e:
        st.error(f"Error in voice recognition: {str(e)}")
        return ""

def load_image(image_path):
    try:
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
            return ImageOps.fit(img, (400, 300))
        return None
    except Exception as e:
        st.warning(f"Couldn't load image: {str(e)}")
        return None

# ========== Meal Planner Functions ==========
def generate_shopping_list():
    if 'meal_plan' not in st.session_state:
        st.session_state.shopping_list = []
        return
    
    all_ingredients = []
    for day, meals in st.session_state.meal_plan.items():
        for meal, recipe in meals.items():
            if recipe and isinstance(recipe, dict) and 'cleaned_ingredients' in recipe:
                ingredients = [ing.strip() for ing in recipe["cleaned_ingredients"].split(",")]
                all_ingredients.extend(ingredients)
    
    counted = Counter(all_ingredients)
    st.session_state.shopping_list = [f"{v}× {k}" for k, v in counted.items()]

def calculate_nutrition_totals():
    if 'meal_plan' not in st.session_state:
        return
    
    totals = {
        "Calories": 0,
        "Protein": 0,
        "Carbs": 0,
        "Fats": 0,
        "Fiber": 0
    }
    
    for day, meals in st.session_state.meal_plan.items():
        for meal, recipe in meals.items():
            if recipe and isinstance(recipe, dict):
                nutrition = fetch_nutritional_info(recipe["title"])
                for key in totals:
                    num = ''.join(filter(str.isdigit, nutrition[key]))
                    if num:
                        totals[key] += int(num)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Calories", f"{totals['Calories']} kcal", delta_color="off")
    with col2:
        st.metric("Protein", f"{totals['Protein']}g", delta_color="off")
    with col3:
        st.metric("Carbs", f"{totals['Carbs']}g", delta_color="off")
    
    col4, col5 = st.columns(2)
    with col4:
        st.metric("Fats", f"{totals['Fats']}g", delta_color="off")
    with col5:
        st.metric("Fiber", f"{totals['Fiber']}g", delta_color="off")

# ========== Recipe Details Page ==========
def show_recipe_details(recipe):
    st.title(recipe["title"])
    
    # Recipe badges
    st.write(f"""
        **Cuisine:** {recipe.get('cuisine', 'Unknown')} | 
        **Diet:** {recipe.get('diet', 'Regular')} | 
        **Time:** {random.randint(10, 60)} mins
    """)
    
    # Recipe image and links
    img_col, link_col = st.columns([3, 1])
    with img_col:
        img = load_image(recipe.get("image_path"))
        if img:
            st.image(img, width=600, caption=recipe["title"], use_column_width='auto')
    
    with link_col:
        st.subheader("Recipe Links")
        google_search_url = f"https://www.google.com/search?q={recipe['title'].replace(' ', '+')}+recipe"
        youtube_search_url = f"https://www.youtube.com/results?search_query={recipe['title'].replace(' ', '+')}+recipe"
        allrecipes_url = f"https://www.allrecipes.com/search?q={recipe['title'].replace(' ', '+')}"
        
        st.markdown(f"[🔍 Google Recipe]({google_search_url})")
        st.markdown(f"[▶️ YouTube Tutorial]({youtube_search_url})")
        st.markdown(f"[🥘 AllRecipes]({allrecipes_url})")

    # Main content columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Ingredients
        st.subheader("Ingredients")
        ingredients = [ing.strip() for ing in recipe["cleaned_ingredients"].split(",")]
        for ing in ingredients:
            st.markdown(f"- {ing}")
        
        # Nutritional info
        st.subheader("Nutritional Info")
        nutrition = fetch_nutritional_info(recipe["title"])
        for key, value in nutrition.items():
            st.markdown(f"**{key}:** {value}")
    
    with col2:
        # Instructions
        st.subheader("Instructions")
        instructions = recipe["instructions"]
        if any(char.isdigit() for char in instructions[:50]):
            steps = [step.strip() for step in instructions.split('\n') if step.strip()]
            for i, step in enumerate(steps, 1):
                st.markdown(f"**Step {i}:** {step}")
        else:
            st.markdown(instructions)
    
    # Rating system
    st.subheader("Rate this Recipe")
    rating = st.slider("Your rating (1-5 stars)", 1, 5, 3, key=f"rating_{recipe['title']}")
    st.write(f"{'★' * rating}{'☆' * (5 - rating)}")
    
    if st.button("Submit Rating", key=f"rate_{recipe['title']}", type="primary"):
        st.session_state.ratings[recipe["title"]] = rating
        st.success("Thanks for your rating!")
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("❤️ Add to Wishlist", key=f"wish_{recipe['title']}"):
            if 'wishlist' not in st.session_state:
                st.session_state.wishlist = []
            
            if not any(r['title'] == recipe['title'] for r in st.session_state.wishlist):
                st.session_state.wishlist.append(recipe)
                st.success("Added to Wishlist!")
            else:
                st.warning("Recipe already in wishlist!")
    
    with col2:
        if st.button("← Back to Results", key=f"back_{recipe['title']}"):
            st.session_state.selected_recipe = None
            st.rerun()
    
    with col3:
        if st.button("⏱️ Cooking Timer", key=f"timer_{recipe['title']}"):
            st.session_state.show_timer = True
    
    with col4:
        if st.button("🛒 Add to Shopping List", key=f"shop_{recipe['title']}"):
            if 'shopping_list' not in st.session_state:
                st.session_state.shopping_list = []
            
            ingredients = [ing.strip() for ing in recipe["cleaned_ingredients"].split(",")]
            st.session_state.shopping_list.extend(ingredients)
            st.success(f"Added {len(ingredients)} items to shopping list!")
    
    # Cooking timer
    if st.session_state.get('show_timer', False):
        st.subheader("Cooking Timer")
        minutes = st.number_input("Set timer (minutes)", min_value=1, max_value=120, value=15, 
                                key=f"timer_min_{recipe['title']}")
        
        if st.button("Start Timer", key=f"start_timer_{recipe['title']}", type="primary"):
            with st.empty():
                progress_bar = st.progress(0)
                for secs in range(minutes*60, 0, -1):
                    mins, sec = divmod(secs, 60)
                    progress = 1 - (secs / (minutes*60))
                    st.markdown(f"⏳ {mins:02d}:{sec:02d}")
                    progress_bar.progress(progress)
                    time.sleep(1)
                st.success("⏰ Time's up! Enjoy your meal!")

# ========== Recipe Suggestions Page ==========
def recipe_suggestions_page():
    st.title("Find Your Perfect Recipe")
    
    with st.expander("🔍 Advanced Search Filters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            cuisine_type = st.selectbox("Select Cuisine Type", 
                                      ["Any"] + sorted(df["cuisine"].dropna().unique()))
            diet_type = st.selectbox("Select Diet Type", 
                        ["Any", "Vegetarian", "Vegan", "Non-Vegetarain"])
            meal_type = st.selectbox("Select Meal Type", 
                                    ["Any", "Breakfast", "Lunch", "Dinner", "Snack", "Dessert"])
            
        with col2:
            taste_profile = st.selectbox("Select Taste Profile", 
                                       ["Any", "Sweet", "Savory", "Spicy",])
            
    
    st.subheader("Search by Ingredients")
    input_method = st.radio("Input Method:", ["Type", "Voice"], horizontal=True)
    
    all_possible_ingredients = list(set(
        ing.strip().lower()
        for ing_list in df["cleaned_ingredients"].dropna()
        for ing in ing_list.split(",")
    ))

    ingredients_input = ""
    
    if input_method == "Type":
        user_input = st.text_input("Enter ingredients (comma-separated):", "", 
                                 placeholder="e.g., chicken, rice, tomatoes")
        if user_input:
            suggested = smart_ingredient_suggestions(user_input, all_possible_ingredients)
            if suggested:
                st.caption(f"💡 Suggestions: {', '.join(suggested)}")
        ingredients_input = user_input
    else:
        if st.button("🎤 Start Voice Input"):
            voice_text = get_voice_input()
            ingredients_input = st.text_input("Edit voice input:", voice_text) if voice_text else ""
    
    if "selected_recipe" not in st.session_state:
        st.session_state.selected_recipe = None
    if "recipe_options" not in st.session_state:
        st.session_state.recipe_options = None
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Search Recipes", use_container_width=True, type="primary"):
            if ingredients_input:
                ingredients = [ing.strip().lower() for ing in ingredients_input.split(",") if ing.strip()]
                filtered_df = filter_recipe(cuisine_type, diet_type, meal_type, taste_profile)
                
                matching_recipes = filtered_df[
                    filtered_df["cleaned_ingredients"].apply(
                        lambda x: all(ing in x.lower() for ing in ingredients)
                    )
                ]
                
                if not matching_recipes.empty:
                    st.session_state.recipe_options = matching_recipes.sample(min(6, len(matching_recipes))).to_dict('records')
                    st.session_state.selected_recipe = None
                    st.success(f"Found {len(matching_recipes)} matching recipes!")
                else:
                    st.error("No recipes found matching your filters and ingredients.")
            else:
                st.error("Please input some ingredients first.")
    
    with col2:
        if st.button("🎲 Surprise Me!", use_container_width=True):
            filtered_df = filter_recipe(cuisine_type, diet_type, meal_type, taste_profile)
            
            if not filtered_df.empty:
                st.session_state.recipe_options = filtered_df.sample(min(6, len(filtered_df))).to_dict('records')
                st.session_state.selected_recipe = None
                st.success("Here are some surprise recipes!")
            else:
                st.error("No recipes available for surprise with these filters.")
    
    if st.session_state.recipe_options is not None and st.session_state.selected_recipe is None:
        st.subheader("Recommended Recipes")
        
        cols = st.columns(3)
        for idx, recipe in enumerate(st.session_state.recipe_options):
            with cols[idx % 3]:
                with st.container():
                    img = load_image(recipe.get("image_path"))
                    if img:
                        st.image(img, use_column_width=True)
                    
                    st.subheader(recipe['title'])
                    
                    st.write(f"""
                        **Cuisine:** {recipe.get('cuisine', 'Unknown')} | 
                        **Time:** {random.randint(10, 60)} mins
                    """)
                    
                    if st.button("View Recipe", key=f"view_{recipe['title']}", use_container_width=True):
                        st.session_state.selected_recipe = recipe
                        st.rerun()
    
    if st.session_state.selected_recipe is not None:
        recipe = st.session_state.selected_recipe
        show_recipe_details(recipe)

# ========== Meal Planner Page ==========
def meal_planner_page():
    st.title("🗓️ Weekly Meal Planner")
    
    # Initialize meal plan
    if 'meal_plan' not in st.session_state:
        st.session_state.meal_plan = {
            day: {meal: None for meal in ["Breakfast", "Lunch", "Dinner", "Snacks"]} 
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        }
    
    # Display planner grid with tabs for better mobile experience
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tabs = [tab1, tab2, tab3, tab4, tab5, tab6, tab7]
    meals = ["Breakfast", "Lunch", "Dinner", "Snacks"]
    
    for i, (day, tab) in enumerate(zip(days, tabs)):
        with tab:
            st.subheader(day)
            
            for meal in meals:
                with st.expander(f"{meal}", expanded=True):
                    current_recipe = st.session_state.meal_plan[day][meal]
                    
                    if current_recipe:
                        # Display recipe card
                        st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                                <h4>{current_recipe['title']}</h4>
                                <p>⏱️ {random.randint(10, 60)} mins | 🏷️ {current_recipe.get('cuisine', 'Unknown')}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button(f"❌ Remove {meal}", key=f"remove_{day}_{meal}"):
                                st.session_state.meal_plan[day][meal] = None
                                st.rerun()
                        with col2:
                            if st.button(f"👀 View Recipe", key=f"view_{day}_{meal}"):
                                st.session_state.selected_recipe = current_recipe
                                st.session_state.current_page = "Find Recipes"
                                st.rerun()
                    else:
                        st.info(f"No {meal.lower()} planned yet")
                    
                    # Get wishlist recipes for dropdown
                    wishlist_titles = ["Select recipe..."] + [r["title"] for r in st.session_state.get('wishlist', [])]
                    selected = st.selectbox(
                        f"Choose {meal} recipe",
                        wishlist_titles,
                        key=f"select_{day}_{meal}",
                        label_visibility="collapsed"
                    )
                    
                    if selected != "Select recipe...":
                        selected_recipe = next(r for r in st.session_state.wishlist if r["title"] == selected)
                        st.session_state.meal_plan[day][meal] = selected_recipe
                        st.rerun()
    
    # Action buttons
    if st.button("🛒 Generate Shopping List", type="primary", use_container_width=True):
        generate_shopping_list()
        st.session_state.current_page = "Shopping List"
        st.rerun()

# ========== Shopping List Page ==========
def shopping_list_page():
    st.title("Shopping List")
    
    # Generate if not exists
    if 'shopping_list' not in st.session_state:
        generate_shopping_list()
    
    # Categorize items
    categories = {
        "Produce": [],
        "Dairy": [],
        "Meat/Fish": [],
        "Pantry": [],
        "Spices": [],
        "Other": []
    }
    
    for item in st.session_state.get('shopping_list', []):
        item_lower = item.lower()
        if any(x in item_lower for x in ["milk", "cheese", "yogurt", "butter", "cream"]):
            categories["Dairy"].append(item)
        elif any(x in item_lower for x in ["chicken", "beef", "fish", "pork", "meat", "lamb"]):
            categories["Meat/Fish"].append(item)
        elif any(x in item_lower for x in ["tomato", "onion", "spinach", "potato", "carrot", "vegetable", "fruit", "apple", "banana"]):
            categories["Produce"].append(item)
        elif any(x in item_lower for x in ["salt", "pepper", "spice", "cumin", "cinnamon", "herb"]):
            categories["Spices"].append(item)
        elif any(x in item_lower for x in ["rice", "pasta", "flour", "sugar", "oil", "bread"]):
            categories["Pantry"].append(item)
        else:
            categories["Other"].append(item)
    
    # Display categorized list
    for category, items in categories.items():
        if items:
            with st.expander(f"{category} ({len(items)})", expanded=True):
                for item in items:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.checkbox("", key=f"check_{item}", value=False)
                    with col2:
                        st.write(item)
    
    # Manual add item
    with st.form("add_item_form"):
        new_item = st.text_input("Add custom item")
        if st.form_submit_button("Add"):
            if new_item:
                st.session_state.shopping_list.append(new_item)
                st.rerun()
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download List",
            "\n".join(st.session_state.shopping_list),
            file_name="shopping_list.txt",
            use_container_width=True
        )
    with col2:
        if st.button("🗑️ Clear List", use_container_width=True):
            st.session_state.shopping_list = []
            st.rerun()

# ========== Wishlist Page ==========
def wishlist_page():
    st.title("Your Wishlist")
    
    if not st.session_state.get('wishlist', []):
        st.info("Your wishlist is empty. Save recipes from the Find Recipes page!")
        return
    
    cols = st.columns(3)
    for idx, recipe in enumerate(st.session_state.wishlist):
        with cols[idx % 3]:
            with st.container():
                img = load_image(recipe.get("image_path"))
                if img:
                    st.image(img, use_column_width=True)
                
                st.subheader(recipe['title'])
                
                st.write(f"""
                    **Cuisine:** {recipe.get('cuisine', 'Unknown')} | 
                    **Time:** {random.randint(10, 60)} mins
                """)
                
                if st.button("View Recipe", key=f"view_wish_{recipe['title']}", use_container_width=True):
                    st.session_state.selected_recipe = recipe
                    st.session_state.current_page = "Find Recipes"
                    st.rerun()
                
                if st.button("❌ Remove", key=f"remove_{recipe['title']}", use_container_width=True):
                    st.session_state.wishlist = [r for r in st.session_state.wishlist if r['title'] != recipe['title']]
                    st.rerun()

# ========== Home Page ==========
def home_page():
    st.title("Welcome to AI Recipe Assistant!")
    st.markdown("""
        ## Get started by:
        1. Searching for recipes based on ingredients
        2. Saving your favorites to your wishlist
        3. Generating shopping lists
        4. Planning your weekly meals
        
        👈 **Use the sidebar** to navigate to different features!
    """)

# ========== About Page ==========
def about_page():
    st.title("About RecipeAI")
    st.markdown("""
        ### 🍳 RecipeAI - Your Personal Cooking Assistant
        
        RecipeAI helps you discover new recipes, plan your meals, and organize your shopping lists - all in one place!
        
        **Features:**
        - 🕵️‍♀️ Find recipes based on ingredients, cuisine, diet, and more
        - ❤️ Save your favorite recipes to your wishlist
        - 🗓️ Plan your weekly meals
        - 🛒 Generate smart shopping lists
        - ⏱️ Cooking timers to help you stay on track
        
        **How to use:**
        1. Search for recipes using the "Find Recipes" page
        2. Save recipes you like to your wishlist
        3. Plan your meals for the week using the "Meal Planner"
        4. Generate a shopping list based on your meal plan
        
        **About the developer:**
        This app was created by [Your Name] to help home cooks discover new recipes and simplify meal planning.
        
        For feedback or suggestions, please contact [your email].
    """)

# ========== Main App ==========
def main():
    # Initialize session state
    if "wishlist" not in st.session_state:
        st.session_state.wishlist = []
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "selected_recipe" not in st.session_state:
        st.session_state.selected_recipe = None
    if "recipe_options" not in st.session_state:
        st.session_state.recipe_options = None
    if "meal_plan" not in st.session_state:
        st.session_state.meal_plan = {
            day: {meal: None for meal in ["Breakfast", "Lunch", "Dinner", "Snacks"]} 
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        }
    if "shopping_list" not in st.session_state:
        st.session_state.shopping_list = []
    if "show_timer" not in st.session_state:
        st.session_state.show_timer = False
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2>🍳 RecipeAI</h2>
                <p>Your personal recipe assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=["Home", "Find Recipes", "Wishlist", "Meal Planner", "Shopping List", "About"],
            icons=['house', 'search', 'heart', 'calendar', 'cart', 'info-circle'],
            default_index=["Home", "Find Recipes", "Wishlist", "Meal Planner", "Shopping List", "About"].index(
                st.session_state.current_page),
            menu_icon="cast",
            styles={
                "container": {"padding": "0!important"},
                "icon": {"color": "#ff4b4b", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background-color": "#ff4b4b"},
            }
        )
        
        st.session_state.current_page = selected
        
        st.markdown("---")
        st.markdown("""
            ### Your Stats
            - ❤️ **Wishlisted Recipes:** {wishlist_count}
            - ⭐ **Rated Recipes:** {rated_count}
            - 🗓️ **Meals Planned:** {meals_planned}
            - 🛒 **Shopping Items:** {shopping_items}
        """.format(
            wishlist_count=len(st.session_state.wishlist),
            rated_count=len(st.session_state.ratings),
            meals_planned=sum(1 for day in st.session_state.meal_plan.values() 
                            for meal in day.values() if meal is not None),
            shopping_items=len(st.session_state.shopping_list)
        ))
        
        if st.button("Clear All Data", type="primary", use_container_width=True):
            st.session_state.wishlist = []
            st.session_state.ratings = {}
            st.session_state.meal_plan = {
                day: {meal: None for meal in ["Breakfast", "Lunch", "Dinner", "Snacks"]} 
                for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            }
            st.session_state.shopping_list = []
            st.success("All data cleared!")
            st.rerun()
    
    # Display current page
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Find Recipes":
        recipe_suggestions_page()
    elif st.session_state.current_page == "Wishlist":
        wishlist_page()
    elif st.session_state.current_page == "Meal Planner":
        meal_planner_page()
    elif st.session_state.current_page == "Shopping List":
        shopping_list_page()
    elif st.session_state.current_page == "About":
        about_page()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page or try again later.")
