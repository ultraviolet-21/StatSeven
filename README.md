# StatSeven

StatSeven is a Python-based project for exploring per-game stats for any NHL team and custom playoff matchups between any two teams. Includes a Machine Learning model that predicts the score of hypothetical games based on regular season per-game stats (faceoff %, shots on goal, etc.) Combines reusable core functions with interactive notebook-based analysis.

## Inspiration

While most sports prediction/ betting sites give odds, these are based on the predictions other users make, rather than regular season per-game stats. Furthermore, they usually don't allow users to experiment with custom matchups. With the NHL and NBA playoffs starting soon, I originally intended to build parallel logic that enables users to predict outcomes in either sport. However, during development, I noticed basketball-reference.com had anti-bot measures that would require heavy browser automation beyong the scope of this project. The corresponding hockey-reference.com allowed for more lightweight scraping, so I developed an NHL-focused program.

## Project Structure

 - `src/fetch_gamelog.py` - functions to fetch a team's gamelog from hockey-reference.com and refine it
 - `src/predict_result.py` - ML model and functions to simulate playoff series
 - `notebooks/explore_gamelog.ipynb` - interactive exploration of any team's gamelog
 - `notebooks/compare_stats.ipynb` - additional experiments, including simulating 7-game playoff series

### Dependencies
 - requests (2.33.1)
 - numpy (2.4.4)
 - pandas (3.0.2)
 - matplotlib (3.10.8)
 - scikit-learn (model_selection, linear_model, preprocessing) (1.8.0)

## Getting Started

1. Open VSCode and clone the repository.

2. Set up Python virtual environment and install dependencies.

3. Install Python and Jupyter packages for VSCode if not already installed. Then select a kernel.

4. Open either Jupyter notebook and replace empty strings with team abbreviations!
