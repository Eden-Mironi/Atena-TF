#!/usr/bin/env python3
"""
Create Google Colab-ready versions of ATENA-TF notebooks
"""

import json
from pathlib import Path


# Welcome Notebook
welcome_nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# 🚀 ATENA-TF Welcome - Google Colab Edition\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eden-Mironi/Atena-TF/blob/main/Notebooks/ATENA_TF_Welcome_Colab.ipynb)\n\n**Quick Start**: Click \"Runtime\" → \"Run all\" and wait 2-3 minutes!\n\n---"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Setup: Detect environment\nimport sys\nIN_COLAB = 'google.colab' in sys.modules\nprint(f\"Running in {'Colab' if IN_COLAB else 'local'} mode\")"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Setup: Install dependencies\nif IN_COLAB:\n    !pip install -q tensorflow>=2.16.0 tensorflow-probability>=0.24.0\n    !pip install -q gym==0.25.2 'numpy<2.0.0' pandas matplotlib seaborn scipy\n    !pip install -q cachetools zss nltk openpyxl\n    print(\"✅ Dependencies installed!\")"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Setup: Get project files\nif IN_COLAB:\n    from google.colab import files\n    import zipfile, os\n    \n    # Option A: Clone from GitHub (recommended)\n    if not os.path.exists('/content/atena-tf'):\n        print(\"📥 Cloning from GitHub...\")\n        !git clone https://github.com/Eden-Mironi/Atena-TF.git /content/atena-tf\n        print(\"✅ Cloned successfully!\")\n    \n    # Option B: Upload ZIP (uncomment if GitHub fails)\n    # print(\"📤 Upload your 'atena-tf.zip' file:\")\n    # uploaded = files.upload()\n    # if uploaded:\n    #     zip_name = list(uploaded.keys())[0]\n    #     with zipfile.ZipFile(zip_name) as z:\n    #         z.extractall('/content')\n    #     dirs = [f for f in os.listdir('/content') if 'atena' in f.lower()]\n    #     if dirs:\n    #         os.rename(f'/content/{dirs[0]}', '/content/atena-tf')\n    \n    %cd /content/atena-tf\n    !pip install -q -e .\n    print(\"✅ Project ready!\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "---\n\n# Welcome to ATENA-TF!\n\nAI-Powered Data Exploration using Deep Reinforcement Learning\n\n### Key Features\n- 🤖 Automated EDA\n- 👤 Human-like exploration\n- 💡 Interactive recommendations\n- 📊 Multi-dataset support"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Example 1: Create Environment"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Setup\nimport numpy as np\nif not hasattr(np, \"bool8\"):\n    np.bool8 = np.bool_\n\nimport gym\nimport pandas as pd\n\n# Create environment\nenv = gym.make('ATENAcont-v0')\nobs = env.reset(dataset_number=0)\n\nprint(f\"✅ Environment created!\")\nprint(f\"📊 Observation: {obs.shape}\")\nprint(f\"📋 Dataset: {env.data.shape}\")\ndisplay(env.data.head(10))"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Example 2: Manual Action"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Take a filter action\naction = np.array([1, 1, 0, 0.5, 0, 0])\nenv.render()\n\nobs, reward, done, info = env.step(action)\n\nprint(f\"Action: {info['action']}\")\nprint(f\"Reward: {reward:.2f}\")\n\nif info.get('raw_display'):\n    df = info['raw_display'][1] or info['raw_display'][0]\n    display(df.head(10))\n    print(f\"New shape: {df.shape}\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "---\n\n## 🎉 Success!\n\nYou've run ATENA-TF in Google Colab!\n\n### Next Steps\n- Try the evaluation notebook\n- Read `README.md` for documentation\n- Train your own agent locally\n\n**Questions?** Contact your professor! 🚀"
        }
    ],
    "metadata": {
        "colab": {"name": "ATENA-TF Welcome (Colab)", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}


# Live Recommendations Notebook
live_nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "# 🤖 ATENA-TF Live Recommendations - Colab Edition\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eden-Mironi/Atena-TF/blob/main/Notebooks/Live_Recommendations_System_Colab.ipynb)\n\n**Interactive AI-Powered Data Exploration**\n\nClick \"Runtime\" → \"Run all\" to start!\n\n---\n\n## How it works:\n1. 🧠 Trained model suggests next best action\n2. 👤 You decide: follow AI or choose your own\n3. 📊 See instant results\n4. 🔄 Repeat and explore!"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Setup\nimport sys\nIN_COLAB = 'google.colab' in sys.modules\nprint(f\"Running in {'Colab' if IN_COLAB else 'local'} mode\")"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "if IN_COLAB:\n    !pip install -q tensorflow>=2.16.0 tensorflow-probability>=0.24.0\n    !pip install -q gym==0.25.2 'numpy<2.0.0' pandas matplotlib seaborn scipy\n    !pip install -q cachetools zss nltk openpyxl ipywidgets\n    print(\"✅ Dependencies installed!\")"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "if IN_COLAB:\n    import os\n    \n    # Clone from GitHub\n    if not os.path.exists('/content/atena-tf'):\n        print(\"📥 Cloning from GitHub...\")\n        !git clone https://github.com/Eden-Mironi/Atena-TF.git /content/atena-tf\n        print(\"✅ Cloned!\")\n    \n    %cd /content/atena-tf\n    !pip install -q -e .\n    print(\"✅ Ready!\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "---\n\n## Initialize Recommender System"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Import recommender components\nimport ipywidgets as widgets\nfrom IPython.display import display, clear_output\nimport pandas as pd\nimport numpy as np\n\nif not hasattr(np, \"bool8\"):\n    np.bool8 = np.bool_\n\nfrom live_recommender_agent import TFRecommenderAgent, find_latest_trained_model\n\nprint(\"✅ Imports successful!\")"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Load trained model\nmodel_path = find_latest_trained_model()\n\nif model_path:\n    print(f\"✅ Found model: {model_path}\")\n    \n    recommender_agent = TFRecommenderAgent(\n        model_path=model_path,\n        dataset_number=0,\n        schema='NETWORKING'\n    )\n    \n    print(\"\\n🎉 Recommender initialized!\")\n    print(f\"📊 Dataset shape: {recommender_agent.original_dataset.shape}\")\n    print(f\"📋 Columns: {list(recommender_agent.original_dataset.columns)}\")\nelse:\n    print(\"⚠️ No trained model found\")\n    print(\"Please train first: python main.py\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 🎛️ Interactive Control Panel"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Create UI controls\noutput_area = widgets.Output()\n\nrecommendation_display = widgets.HTML(\n    value=\"<b><font color='blue'>🤖 Getting recommendation...</font></b>\"\n)\n\ndef show_recommendation():\n    try:\n        rec = recommender_agent.get_agent_action_str()\n        recommendation_display.value = f\"<b><font color='red'>🤖 AI: {rec}</font></b>\"\n    except Exception as e:\n        recommendation_display.value = f\"<b><font color='orange'>⚠️ Error: {e}</font></b>\"\n\nif 'recommender_agent' in locals():\n    show_recommendation()\n\nprint(\"✅ UI created!\")"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Action controls\naction_type = widgets.Dropdown(\n    options=['Group', 'Filter', 'Back'],\n    description='Action:'\n)\n\ncolumn_select = widgets.Dropdown(\n    options=['packet_number', 'eth_src', 'eth_dst', 'tcp_srcport', 'tcp_dstport'],\n    description='Column:'\n)\n\napply_ai_btn = widgets.Button(\n    description='✅ Apply AI Recommendation',\n    button_style='success'\n)\n\napply_custom_btn = widgets.Button(\n    description='🎯 Apply My Action',\n    button_style='info'\n)\n\nreset_btn = widgets.Button(\n    description='🔄 Reset',\n    button_style='warning'\n)\n\nprint(\"✅ Controls ready!\")"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Button handlers\ndef apply_ai_handler(btn):\n    with output_area:\n        clear_output(wait=True)\n        try:\n            result = recommender_agent.apply_agent_action(use_last_recommendation=True)\n            if result.df_to_display is not None:\n                print(f\"✅ Result ({len(result.df_to_display)} rows):\")\n                display(result.df_to_display.head(10))\n        except Exception as e:\n            print(f\"❌ Error: {e}\")\n    show_recommendation()\n\ndef reset_handler(btn):\n    with output_area:\n        clear_output(wait=True)\n        try:\n            recommender_agent.reset_environment()\n            print(\"✅ Reset complete!\")\n            display(recommender_agent.original_dataset.head(10))\n        except Exception as e:\n            print(f\"❌ Error: {e}\")\n    show_recommendation()\n\napply_ai_btn.on_click(apply_ai_handler)\nreset_btn.on_click(reset_handler)\n\nprint(\"✅ Handlers ready!\")"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## 🎮 Live Interface"
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": "# Display interface\nui = widgets.VBox([\n    widgets.HTML(\"<h3>🎛️ Control Panel</h3>\"),\n    recommendation_display,\n    widgets.HTML(\"<hr>\"),\n    widgets.HBox([action_type, column_select]),\n    widgets.HBox([apply_ai_btn, apply_custom_btn, reset_btn]),\n    widgets.HTML(\"<hr>\"),\n    widgets.HTML(\"<h3>📊 Results</h3>\"),\n    output_area\n])\n\ndisplay(ui)\n\n# Show initial data\nif 'recommender_agent' in locals():\n    with output_area:\n        print(\"🗂️ Original Dataset:\")\n        display(recommender_agent.original_dataset.head(10))"
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "---\n\n## 🎉 Success!\n\n### Usage:\n1. **AI Recommendation** shown above\n2. **Click buttons** to apply actions\n3. **See results** in output area\n4. **Repeat** to explore!\n\n### Tips:\n- Follow AI to see what it learned\n- Try your own actions\n- Use Reset to start over\n\n**Enjoy exploring with AI! 🚀**"
        }
    ],
    "metadata": {
        "colab": {"name": "Live Recommendations (Colab)", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}


# Save notebooks
output_dir = Path("Notebooks")
output_dir.mkdir(exist_ok=True)

welcome_path = output_dir / "ATENA_TF_Welcome_Colab.ipynb"
with open(welcome_path, 'w') as f:
    json.dump(welcome_nb, f, indent=2)
print(f"✅ Created: {welcome_path}")

eval_path = output_dir / "Live_Recommendations_System_Colab.ipynb"
with open(eval_path, 'w') as f:
    json.dump(live_nb, f, indent=2)
print(f"✅ Created: {eval_path}")

print("\n🎉 Colab-ready notebooks created successfully!")
print("\n📝 Next steps:")
print("1. Upload these notebooks to Google Colab")
print("2. Test them by clicking 'Runtime' → 'Run all'")
print("3. Share the links with your professor")
print("\n📓 Notebooks created:")
print("   • ATENA_TF_Welcome_Colab.ipynb - Quick demo")
print("   • Live_Recommendations_System_Colab.ipynb - Interactive exploration")
