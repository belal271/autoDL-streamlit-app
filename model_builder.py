import streamlit as st
import pandas as pd
import numpy as np
import os
from keras import layers, models
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split

# Initialize session state for preprocessed data
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = None

st.title('Deep Learning Model Builder')

# Load existing data if available
if os.path.exists("sourcedata.csv"):
    st.session_state.df = pd.read_csv("sourcedata.csv", index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Welcome to the autoML project!")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This project helps you build a model for your data without using a single line of code!")

# Upload page functionality
if choice == "Upload":
    st.title("Upload your file for modelling")
    file = st.file_uploader("Upload your file here")
    if file:
        st.session_state.df = pd.read_csv(file, index_col=None)
        st.session_state.df.to_csv("sourcedata.csv", index=None)
        st.write("Original Data:")
        st.dataframe(st.session_state.df)
        
        if st.button("Preprocess Data"):
            st.session_state.preprocessed_df = preprocess_data(st.session_state.df)
            st.session_state.preprocessed = True
            st.success("Data has been preprocessed successfully!")
            st.write("Preview of preprocessed data:")
            st.dataframe(st.session_state.preprocessed_df)

# Profiling page functionality
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if st.session_state.df is not None:
        if st.session_state.preprocessed:
            data_option = st.radio("Select data to profile:", ["Original Data", "Preprocessed Data"])
            if data_option == "Preprocessed Data":
                profile_df = st.session_state.preprocessed_df.profile_report()
            else:
                profile_df = st.session_state.df.profile_report()
        else:
            profile_df = st.session_state.df.profile_report()
        st_profile_report(profile_df)
    else:
        st.warning("Please upload data first.")

# Modelling page functionality
if choice == "Modelling":
    st.title("Build the model")
    
    if not st.session_state.preprocessed:
        st.warning("Please preprocess your data first in the Upload section before proceeding with modeling.")
        st.stop()
    
    df = st.session_state.preprocessed_df
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    
    if chosen_target not in df.columns:
        st.error(f"'{chosen_target}' column not found in the data.")
    else:
        # Determine if it's a classification or regression problem
        unique_values = df[chosen_target].nunique()
        is_classification = unique_values < 10  # Assume classification if less than 10 unique values
        
        st.info(f"Detected problem type: {'Classification' if is_classification else 'Regression'}")
        
        # Architecture selection
        st.header('Model Configuration')
        
        col1, col2 = st.columns(2)
        with col1:
            num_layers = st.number_input('Number of Dense Layers', min_value=1, max_value=10, value=3)
            epochs = st.number_input('Number of Epochs', min_value=1, max_value=1000, value=100)
            batch_size = st.number_input('Batch Size', min_value=8, max_value=256, value=32)
            
        with col2:
            learning_rate = st.select_slider(
                'Learning Rate',
                options=[0.1, 0.01, 0.001, 0.0001],
                value=0.001
            )
            patience = st.number_input('Early Stopping Patience', min_value=5, max_value=50, value=10)
        
        st.subheader('Layer Configuration')
        layers_config = []
        for i in range(num_layers):
            col1, col2 = st.columns(2)
            with col1:
                neurons = st.number_input(f'Neurons in Layer {i+1}', 
                                       min_value=1, 
                                       max_value=512, 
                                       value=min(128, df.drop(columns=[chosen_target]).shape[1] * 2))
            with col2:
                activation = st.selectbox(f'Activation for Layer {i+1}', 
                                        ['relu', 'tanh', 'sigmoid'],
                                        index=0)
            layers_config.append({'units': neurons, 'activation': activation})
        
        if st.button("Build Model"):
            # Prepare the data
            X = df.drop(columns=[chosen_target])
            y = df[chosen_target]
            
            # Normalize target for regression
            if not is_classification:
                from sklearn.preprocessing import StandardScaler
                y_scaler = StandardScaler()
                y = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
            
            # Initialize model
            model = models.Sequential()
            
            # Add input layer
            model.add(layers.Dense(layers_config[0]['units'], 
                                 activation=layers_config[0]['activation'],
                                 input_shape=(X.shape[1],)))
            
            # Add hidden layers
            for layer in layers_config[1:]:
                model.add(layers.Dense(layer['units'], activation=layer['activation']))
                model.add(layers.Dropout(0.2))  # Add dropout for regularization
            
            # Add output layer
            if is_classification:
                if unique_values == 2:  # Binary classification
                    model.add(layers.Dense(1, activation='sigmoid'))
                    loss = 'binary_crossentropy'
                    metrics = ['accuracy']
                else:  # Multi-class classification
                    model.add(layers.Dense(unique_values, activation='softmax'))
                    loss = 'sparse_categorical_crossentropy'
                    metrics = ['accuracy']
            else:  # Regression
                model.add(layers.Dense(1, activation='linear'))
                loss = 'mean_squared_error'
                metrics = ['mae']
            
            # Compile model with specified learning rate
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            
            # Save model to session state
            st.session_state.model = model
            st.session_state.is_classification = is_classification
            if not is_classification:
                st.session_state.y_scaler = y_scaler
            
            # Display model summary
            st.subheader("Model Summary")
            with st.expander("Click to see model summary"):
                model.summary(print_fn=lambda x: st.text(x))
            
            # Visualize the model architecture
            st.subheader("Model Visualization")
            try:
                plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
                st.image('model.png')
            except ImportError:
                st.error("You need to install pydot and graphviz for model visualization.")
        
        if st.button("Train Model"):
            if 'model' in st.session_state:
                model = st.session_state.model
                X = df.drop(columns=[chosen_target])
                y = df[chosen_target]
                
                # Normalize target for regression
                if not st.session_state.is_classification:
                    y = st.session_state.y_scaler.transform(y.values.reshape(-1, 1)).ravel()
                
                # Split the data
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Early stopping callback
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True
                )
                
                # Initialize metrics storage
                train_metrics = {'loss': [], 'metric': []}
                val_metrics = {'loss': [], 'metric': []}
                
                # Training progress
                progress_bar = st.progress(0)
                metrics_placeholder = st.empty()
                
                # Training loop
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    verbose=1
                )
                
                # Plot training history
                fig = make_subplots(rows=2, cols=1, 
                                  subplot_titles=('Model Loss', 'Model Metric'))
                
                # Loss plots
                fig.add_trace(
                    go.Scatter(y=history.history['loss'], name="Train Loss"),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(y=history.history['val_loss'], name="Validation Loss"),
                    row=1, col=1
                )
                
                # Metric plots
                if st.session_state.is_classification:
                    fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Train Accuracy'), row=2, col=1)
                    fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Val Accuracy'), row=2, col=1)
                else:
                    fig.add_trace(go.Scatter(y=history.history['mae'], name='Train MAE'), row=2, col=1)
                    fig.add_trace(go.Scatter(y=history.history['val_mae'], name='Val MAE'), row=2, col=1)
                
                fig.update_layout(height=600, title_text='Training Metrics')
                fig.update_xaxes(title_text='Epoch', row=2, col=1)
                if st.session_state.is_classification:
                    fig.update_yaxes(title_text='Accuracy', row=2, col=1)
                else:
                    fig.update_yaxes(title_text='Mean Absolute Error', row=2, col=1)
                
                st.plotly_chart(fig)
                
                # Final metrics
                st.subheader("Final Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Final Training Loss", f"{history.history['loss'][-1]:.4f}")
                    if st.session_state.is_classification:
                        st.metric("Final Training Accuracy", f"{history.history['accuracy'][-1]:.4f}")
                    else:
                        st.metric("Final Training MAE", f"{history.history['mae'][-1]:.4f}")
                with col2:
                    st.metric("Final Validation Loss", f"{history.history['val_loss'][-1]:.4f}")
                    if st.session_state.is_classification:
                        st.metric("Final Validation Accuracy", f"{history.history['val_accuracy'][-1]:.4f}")
                    else:
                        st.metric("Final Validation MAE", f"{history.history['val_mae'][-1]:.4f}")
                
                # Store final metrics in session state
                st.session_state.final_metrics = {
                    'train_loss': history.history['loss'][-1],
                    'val_loss': history.history['val_loss'][-1]
                }
                
                if st.session_state.is_classification:
                    st.session_state.final_metrics.update({
                        'train_acc': history.history['accuracy'][-1],
                        'val_acc': history.history['val_accuracy'][-1]
                    })
                else:
                    st.session_state.final_metrics.update({
                        'train_mae': history.history['mae'][-1],
                        'val_mae': history.history['val_mae'][-1]
                    })
            else:
                st.error("Model has not been built yet.")

# Download page functionality
if choice == "Download":
    st.title("Download Trained Model")
    
    if 'model' not in st.session_state:
        st.warning("No trained model available. Please train a model first.")
    else:
        # Save model to H5 file
        model = st.session_state.model
        model.save('trained_model.h5')
        
        # Read the saved model file
        with open('trained_model.h5', 'rb') as f:
            model_bytes = f.read()
        
        st.success("Model is ready for download!")
        st.download_button(
            label="Download Model (H5 Format)",
            data=model_bytes,
            file_name="trained_model.h5",
            mime="application/octet-stream"
        )
        
        # Add model summary for reference
        st.subheader("Model Architecture Summary")
        with st.expander("Click to view model summary"):
            model.summary(print_fn=lambda x: st.text(x))
            
        # Add training configuration info if available
        st.subheader("Model Configuration")
        st.write("Problem Type:", "Classification" if st.session_state.is_classification else "Regression")
        
        # Display final metrics if available
        if hasattr(st.session_state, 'final_metrics'):
            st.subheader("Final Model Performance")
            metrics = st.session_state.final_metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Loss", f"{metrics['train_loss']:.4f}")
                if 'train_acc' in metrics:
                    st.metric("Training Accuracy", f"{metrics['train_acc']:.4f}")
                else:
                    st.metric("Training MAE", f"{metrics['train_mae']:.4f}")
            with col2:
                st.metric("Validation Loss", f"{metrics['val_loss']:.4f}")
                if 'val_acc' in metrics:
                    st.metric("Validation Accuracy", f"{metrics['val_acc']:.4f}")
                else:
                    st.metric("Validation MAE", f"{metrics['val_mae']:.4f}")
