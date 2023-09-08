

###-------User input------------------------------------------------------------

# Total number of batches needed to cover all samples for the given batch size 
# and epochs
batch_size <- 32

# Set all paths that lead to the images which will be used in training and 
# evaluations
Train_Color_Path <- "~/path/for/the/training/images"
Train_Gray_Path <- "~/path/for/the/training/images"
Train_Binary_Path <- "~/path/for/the/training/images"

Eval_Color_Path <- "~/path/for/the/evaluation/images"
Eval_Gray_Path <- "~/path/for/the/evaluation/images"
Eval_Binary_Path <- "~/path/for/the/evaluation/images"

# Set all paths for the models to be saved to
Lumos_gray_generator_model_path <- "~/path/for/the/gray/model"
Lumos_binary_generator_model_path <- "~/path/for/the/binary/model"

# Set all paths for the models to be saved to
Lumos_gray_path <- "~/path/for/the/gray/model"
Lumos_binary_path <- "~/path/for/the/binary/model"

# Set the last epoch that was completed during training
last_epoch_completed = 1 # File name will indicate in the name if unknown

# Set the accuracy goals
target_training_accuracy <- 0.99995 # e.g., 99.995% accuracy
target_validation_accuracy <- 0.9999 # e.g., 99.95% accuracy

# Set the desired image sizes
image_size <- c(333, 250) # Width and height


###-------Load Libraries--------------------------------------------------------

# Load libraries
if (!require(progress)) {
  install.packages("progress")
  library(progress)
} else
{
  library(progress)
}

if (!require(SpatialPack)) {
  install.packages("SpatialPack")
  library(SpatialPack)
} else
{
  library(SpatialPack)
}

if (!require(keras)) {
  install.packages("keras")
  library(keras)
} else
{
  library(keras)
}

if (!require(tensorflow)) {
  install.packages("tensorflow")
  library(tensorflow)
} else
{
  library(tensorflow)
}

if (!require(magick)) {
  install.packages("magick")
  library(magick)
} else
{
  library(magick)
}

if (!require(ggplot2)) {
  install.packages("ggplot2")
  library(ggplot2)
} else
{
  library(ggplot2)
}

if (!require(gridExtra)) {
  install.packages("gridExtra")
  library(gridExtra)
} else
{
  library(gridExtra)
}


###-------Create needed functions-----------------------------------------------

# Calculate Structural Similarity Index (SSIM) between two images
# Arguments:
#   image1: The first image for SSIM comparison
#   image2: The second image for SSIM comparison
# Returns:
#   The SSIM score indicating the structural similarity between the two images

calculate_ssim <- function(image1, image2) {
  SSIM(image1, image2)
}



# Custom data loader function with optional augmentation
# Arguments:
#   first_image_path: File path to the first image
#   second_image_path: File path to the second image
#   first_image_size: Dimensions of the first image (e.g., c(height, width,
#   channels))
#   second_image_size: Dimensions of the second image (e.g., c(height, width,
#   channels))
#   apply_augmentation: Boolean flag to control whether augmentation is applied
#   (default: TRUE)
# Returns:
#   A list containing the primary and secondary images, possibly augmented

load_augmented_image <-
  function(first_image_path,
           second_image_path,
           first_image_size,
           second_image_size,
           apply_augmentation = TRUE) {
    
    # Load the images
    image_one <- magick::image_read(first_image_path)
    image_two <- magick::image_read(second_image_path)
    
    if (apply_augmentation && runif(1) < 0.5) {
      augmentation_applied <- FALSE
      
      # Apply augmentation until at least one is successfully applied
      while (!augmentation_applied) {
        if (runif(1) < 0.5) {
          # Apply brightness adjustment to the training image
          image_one <- image_modulate(image_one, brightness = 200)
          augmentation_applied <- TRUE
        } else if (runif(1) < 0.5) {
          # Apply horizontal flip to both images
          image_one <- image_flip(image_one)
          image_two <- image_flip(image_two)
          augmentation_applied <- TRUE
        } else if (runif(1) < 0.5) {
          # Apply vertical flip to both images
          image_one <- image_flop(image_one)
          image_two <- image_flop(image_two)
          augmentation_applied <- TRUE
        } else if (runif(1) < 0.5) {
          # Apply Gaussian noise to the training image
          image_one <- image_noise(image_one, noisetype = "gaussian")
          augmentation_applied <- TRUE
        }
        
      }
      
    }
    
    # Convert images to numeric arrays
    image_array_one <- as.numeric(magick::image_data(image_one))
    image_array_two <- as.numeric(magick::image_data(image_two))
    
    # Reshape images to desired dimensions
    reshaped_primary_image <- array_reshape(
      image_array_one,
      dim = c(
        1, 
        first_image_size[1],
        first_image_size[2],
        first_image_size[3])
    )
    
    reshaped_secondary_image <- array_reshape(
      image_array_two,
      dim = c(
        1,
        second_image_size[1],
        second_image_size[2],
        second_image_size[3]
      )
    )
    
    # Return a list of primary and secondary images
    return(list(Primary = reshaped_primary_image, 
                Secondary = reshaped_secondary_image))
  }


###-------Recover previous data-------------------------------------------------

# List of paths to process
List_of_paths <- c(
  "Train_Color",
  "Train_Gray",
  "Train_Binary",
  "Eval_Color",
  "Eval_Gray",
  "Eval_Binary"
)

Temporary_folder <- "Temporary_folder"

# Process each path in the list
for (list_name in List_of_paths) {
  # Create variable names
  path_var_name <- paste0(list_name, "_Path")
  file_var_name <- paste0(list_name)
  resized_var_name <- paste0("Resized_", list_name)
  
  # Update the path variable to the temporary folder
  assign(path_var_name, file.path(get(path_var_name), Temporary_folder))
  
  # List files in the original path
  assign(
    file_var_name,
    list.files(
      get(path_var_name),
      pattern = ".jpg|.jpeg|.png",
      full.names = TRUE,
      ignore.case = TRUE
    )
  )
  
  # Create resized list variable
  assign(resized_var_name, list())
}

total_samples <- length(Train_Gray)

Lumos_gray_generator_checkpoint <- "/Checkpoint_folder/"

Checkpoint_folder_location <- paste0(Lumos_gray_path, 
                                     Lumos_gray_generator_checkpoint)

Metrics_file <- "Gray_metrics.csv"

Gray_metrics_df <- read.csv(paste0(Checkpoint_folder_location, 
                                   Metrics_file))

if (Gray_metrics_df$epoch[length(Gray_metrics_df$epoch)] > 1)
{
  # Create the ggplot graph for accuracy
  plot_accuracy <- ggplot(Gray_metrics_df, aes(x = epoch)) +
    geom_line(aes(y = accuracy, color = "Training Accuracy")) +
    geom_line(aes(y = validation_accuracy, color = "Validation Accuracy")) +
    geom_smooth(aes(y = accuracy, color = "Training Accuracy"),
                method = "loess",
                se = FALSE) +
    geom_smooth(
      aes(y = validation_accuracy, color = "Validation Accuracy"),
      method = "loess",
      se = FALSE
    ) +
    geom_point(
      aes(y = accuracy, color = "Training Accuracy"),
      shape = 1,
      size = 3 * 0.75
    ) +
    geom_point(
      aes(y = validation_accuracy, color = "Validation Accuracy"),
      shape = 2,
      size = 3 * 0.75
    ) +
    scale_color_manual(values = c(
      "Training Accuracy" = "lightblue",
      "Validation Accuracy" = "gold"
    )) +
    labs(x = NULL,
         y = "Accuracy",
         color = "Metrics") +
    theme_minimal() +
    theme(legend.position = "right") +
    guides(color = guide_legend(override.aes = list(shape = c(1, 2))))
  
  # Create the ggplot graph for loss
  plot_loss <- ggplot(Gray_metrics_df, aes(x = epoch)) +
    geom_line(aes(y = loss, color = "Training Loss")) +
    geom_line(aes(y = validation_loss, color = "Validation Loss")) +
    geom_smooth(aes(y = loss, color = "Training Loss"),
                method = "loess",
                se = FALSE) +
    geom_smooth(
      aes(y = validation_loss, color = "Validation Loss"),
      method = "loess",
      se = FALSE
    ) +
    geom_point(aes(y = loss, color = "Training Loss"),
               shape = 1,
               size = 3 * 0.75) +
    geom_point(
      aes(y = validation_loss, color = "Validation Loss"),
      shape = 2,
      size = 3 * 0.75
    ) +
    scale_color_manual(values = c(
      "Training Loss" = "lightblue",
      "Validation Loss" = "gold"
    )) +
    labs(x = "Epoch",
         y = "Loss",
         color = "Metrics") +
    theme_minimal() +
    theme(legend.position = "right") +
    guides(color = guide_legend(override.aes = list(shape = c(1, 2))))
  
  # Stack the accuracy and loss plots vertically
  stacked_plots <-
    grid.arrange(plot_accuracy, plot_loss, ncol = 1)
  
  # Display the stacked plots
  print(stacked_plots)
}

Index_file <- "Index_df.csv"

Index_df <- read.csv(paste0(Checkpoint_folder_location, 
                            Index_file))

# Access the previous index list for images used during training 
Stored_image_indexs <- Index_df$Index

# Access the previous learning rate
learning_rate <- Index_df$learning_rate[1]

# Assign image sizes and channel counts for images 
color_image_size <- gray_image_size <- binary_image_size <- c(image_size[1], 
                                                              image_size[2], 
                                                              3)
gray_image_size[3] <- binary_image_size[3] <- 1

# Create storage vaults for color and grayscale images
storage_vault_color <- vector("list", length(total_samples))
storage_vault_gray <- vector("list", length(total_samples))

pb <- progress_bar$new(total = length(Stored_image_indexs), 
                       format = paste("[:bar] ETA: :eta", 
                                      sep = ""), 
                       clear = TRUE)

# Iterate over the stored image indices
for (x in 1:length(Stored_image_indexs))
{
  # Get the current index from the stored image indices list
  index <- Stored_image_indexs[x]
  
  # Load unaltered color and grayscale images from file paths
  Storage_images <-
    load_augmented_image(Train_Color[index],
                         Train_Gray[index],
                         color_image_size,
                         gray_image_size,
                         apply_augmentation = FALSE)
  
  # Store unaltered color image in the color storage vault
  storage_vault_color[[index]] <- Storage_images$Primary
  
  # Store unaltered grayscale image in the grayscale storage vault
  storage_vault_gray[[index]] <- Storage_images$Secondary
  pb$tick()
}
pb$terminate()


###-------Recreate the model and environment---------------------------------

Lumos_gray_environment <- new.env()

with(Lumos_gray_environment, {
  Lumos_gray_generator_name <- paste0("Lumos_gray_generator_", 
                                      last_epoch_completed, ".h5")
  
  Lumos_gray_generator_model_path <- 
    paste0(Checkpoint_folder_location, Lumos_gray_generator_name)
  
  Lumos_gray_generator_model_path <- 
    path.expand(Lumos_gray_generator_model_path)
  
  Lumos_gray_generator_model <- 
    keras$models$load_model(Lumos_gray_generator_model_path)
  
  # Compile the model
  optimizer_binary <-
    tf$keras$optimizers$Adam(learning_rate = learning_rate)
  
  Lumos_gray_generator_model %>% compile(loss = "mean_squared_error",
                                         optimizer = optimizer_binary, 
                                         metrics = list("accuracy"))  
})


###-------Train the models------------------------------------------------------

# Total number of batches needed to cover all samples for the given batch size 
# and epochs
epochs <- ceiling(total_samples / batch_size) 
total_batches_needed <- ceiling(length(Train_Color)) 

# Define the file paths for saving the models
Lumos_gray_generator_model_path <- path.expand(Lumos_gray_generator_model_path)

with(Lumos_gray_environment, {
  # Training loop
  for (epoch in (last_epoch_completed + 1):epochs) {
    
    message("Epoch: ", epoch, "/", epochs)
    
    # Create the progress bar
    pb <- progress_bar$new(
      total = total_batches_needed,
      format = paste(
        ":current/:total [:bar] ETA: :eta - Accuracy: :training_accuracy - Loss: :training_loss - Validation accuracy: :validation_accuracy - Validation loss: :validation_loss" ,
        sep = ""
      ),
      clear = TRUE
    )
    
    # Define callback for saving best model based on validation loss
    best_loss_model_checkpoint <- callback_model_checkpoint(
      filepath = paste0(Checkpoint_folder_location, 
                        "/best_Lumos_Gray_loss_model.h5"),
      monitor = "val_loss",
      save_best_only = TRUE,
      mode = "min"
    )
    
    # Define callback for saving best model based on validation accuracy
    best_accuracy_model_checkpoint <- callback_model_checkpoint(
      filepath = paste0(Checkpoint_folder_location, 
                        "/best_Lumos_Gray_accuracy_model.h5"),
      monitor = "val_accuracy",
      save_best_only = TRUE,
      mode = "max"
    )
    
    # Learning rate scheduler
    reduce_lr_callback <-
      callback_reduce_lr_on_plateau(
        monitor = "loss",
        factor = 0.1,
        patience = 2,
        verbose = 0,
        mode = "min",
        min_delta = 0.01,
        cooldown = 2,
        min_lr = 0.000025
      )
    
    # Create a callback to log metrics and save model checkpoints
    callbacks <- list(
      reduce_lr_callback,
      best_loss_model_checkpoint,
      best_accuracy_model_checkpoint
    )
    
    # Loop over the total number of batches
    for (i in 1:total_batches_needed) {
      
      # Create a random permutation of sample indices
      random_indices <- sample(total_samples, replace = FALSE)
      
      # Determine the start and end index for the current batch
      start_index <- 1
      end_index <- min(start_index + batch_size - 1, total_samples)
      
      # Get the random indices for the current batch
      batch_indices <- random_indices[start_index:end_index]
      
      # Initialize arrays to store batch data
      batch_images_color <-
        array(
          dim = c(
            end_index - start_index + 1,
            color_image_size[1],
            color_image_size[2],
            color_image_size[3]
          )
        )
      
      batch_images_gray <-
        array(
          dim = c(
            end_index - start_index + 1,
            gray_image_size[1],
            gray_image_size[2],
            gray_image_size[3]
          )
        )
      
      batch_labels <- array(dim = c(end_index - start_index + 1, 1))
      
      for (j in 1:length(batch_indices)) {
        index <- batch_indices[j]
        
        if (!(index %in% Stored_image_indexs)) {
          Stored_image_indexs <- append(Stored_image_indexs, index)
          
          Storage_images <-
            load_augmented_image(Train_Color[index],
                                 Train_Gray[index],
                                 color_image_size,
                                 gray_image_size,
                                 apply_augmentation = FALSE)
          
          # Store unaltered color and grayscale images in the vaults
          storage_vault_color[[index]] <- Storage_images$Primary
          storage_vault_gray[[index]] <- Storage_images$Secondary
        }
        
        Processed_images <-
          load_augmented_image(Train_Color[index],
                               Train_Gray[index],
                               color_image_size,
                               gray_image_size)
        
        # Add images and labels to the batch arrays
        batch_images_color[j, , ,] <- Processed_images$Primary
        batch_images_gray[j, , ,] <- Processed_images$Secondary
        batch_labels[j,] <- index
      }
      
      # Generate a random index
      eval_random_index <- sample(total_samples, 1)
      
      # Get the random color and grayscale image paths
      random_train_image <- Train_Color[eval_random_index]
      random_test_image <- Train_Gray[eval_random_index]
      
      # Read the random color and grayscale images
      eval_color_image <- magick::image_read(random_train_image)
      eval_gray_image <- magick::image_read(random_test_image)
      
      # Convert magick images to arrays
      eval_color_image_array <-
        as.numeric(magick::image_data(eval_color_image))
      eval_gray_image_array <-
        as.numeric(magick::image_data(eval_gray_image))
      
      # Reshape image arrays to match the desired shape
      eval_reshaped_color_image <-
        array_reshape(
          eval_color_image_array,
          dim = c(
            1,
            color_image_size[1],
            color_image_size[2],
            color_image_size[3]
          )
        )
      
      eval_reshaped_gray_image <-
        array_reshape(
          eval_gray_image_array,
          dim = c(
            1, 
            gray_image_size[1],
            gray_image_size[2],
            gray_image_size[3])
        )
      
      validation_images_color <- eval_reshaped_color_image
      validation_images_gray <- eval_reshaped_gray_image
      
      # Train the model
      history <- fit(
        Lumos_gray_generator_model,
        x = batch_images_color,
        y = batch_images_gray,
        epochs = epochs,
        batch_size = batch_size,
        verbose = 0,
        validation_data = list(validation_images_color, 
                               validation_images_gray),  
        callbacks = callbacks
      )
      
      # Acquire the learning rate in use
      current_lr <- history$metrics$lr[[length(history$metrics$lr)]]
      
      # Generate random indices to sample from storage vaults
      sample_index <- sample(Stored_image_indexs, 1)
      
      sampled_color_images <- storage_vault_color[[sample_index]]
      sampled_gray_images <- storage_vault_gray[[sample_index]]
      
      # Evaluate the generator model on training data
      training_evaluation <- 
        Lumos_gray_generator_model %>% evaluate(x = sampled_color_images,
                                                y = sampled_gray_images,
                                                verbose = 0)
      
      # Evaluate the generator model on testing data
      validation_evaluation <-
        Lumos_gray_generator_model %>% evaluate(x = eval_reshaped_color_image,
                                                y = eval_reshaped_gray_image,
                                                verbose = 0)
      
      # Retrieve the loss from the training evaluation
      training_loss <- training_evaluation["loss"]
      
      # Retrieve the loss from the validation evaluation
      validation_loss <- validation_evaluation["loss"]
      
      # Create empty lists to store images
      training_images_color <- vector("list", length(Eval_Color))
      training_images_gray <- vector("list", length(Eval_Color))
      
      # Generate random indices to sample from storage vaults
      sample_indices <- sample(Stored_image_indexs, length(Eval_Color))
      
      for (x in 1:length(sample_indices))
      {
        index <- sample_indices[x]
        
        sampled_color_images <- storage_vault_color[[index]]
        sampled_gray_images <- storage_vault_gray[[index]]
        
        training_images_color[[x]] <- sampled_color_images
        training_images_gray[[x]] <- sampled_gray_images
      }
      
      training_count = 0
      
      # Loop over the predicted and ground truth grayscale images
      for (i in 1:length(Eval_Gray)) {
        current_prediction <-
          Lumos_gray_generator_model %>% predict(x = training_images_color[[i]], 
                                                 verbose = 0)
        
        # Calculate SSIM between the ground truth grayscale image and the 
        # predicted grayscale image
        ssim_score <-
          calculate_ssim(training_images_gray[[i]], current_prediction)
        
        ssim_value <- ssim_score$SSIM
        
        # Check if the SSIM score is above the similarity threshold (e.g., 90%)
        if (ssim_value >= target_training_accuracy)
        {
          training_count = training_count + 1
        }
      }
      
      # Create empty lists to store images
      eval_images_color <- vector("list", length(Eval_Color))
      eval_images_gray <- vector("list", length(Eval_Color))
      
      # Loop over the samples in Eval_Color
      for (j in 1:length(Eval_Color)) {
        # Read the color image
        color_image <- magick::image_read(Eval_Color[j])
        
        # Convert magick image to array
        color_image_array <-
          as.numeric(magick::image_data(color_image))
        
        # Reshape color image array to match the desired shape
        reshaped_color_image <- array_reshape(
          color_image_array,
          dim = c(
            1,
            color_image_size[1],
            color_image_size[2],
            color_image_size[3]
          )
        )
        
        # Add the reshaped color image to the list
        eval_images_color[[j]] <- reshaped_color_image
        
        # Read the grayscale image
        gray_image <- magick::image_read(Eval_Gray[j])
        
        # Convert magick image to array
        gray_image_array <-
          as.numeric(magick::image_data(gray_image))
        
        # Reshape grayscale image array to match the desired shape
        reshaped_gray_image <- array_reshape(
          gray_image_array,
          dim = c(
            1,
            gray_image_size[1],
            gray_image_size[2],
            gray_image_size[3]
          )
        )
        
        # Add the reshaped grayscale image to the list
        eval_images_gray[[j]] <- reshaped_gray_image
      }
      
      validation_count = 0
      
      # Loop over the predicted and ground truth grayscale images
      for (i in 1:length(Eval_Gray)) {
        current_prediction <-
          Lumos_gray_generator_model %>% predict(x = eval_images_color[[i]], 
                                                 verbose = 0)
        
        # Calculate SSIM between the ground truth grayscale image and the 
        # predicted grayscale image
        ssim_score <-
          calculate_ssim(eval_images_gray[[i]], current_prediction)
        
        ssim_value <- ssim_score$SSIM
        
        # Check if the SSIM score is above the similarity threshold (e.g., 90%)
        if (ssim_value >= target_validation_accuracy)
        {
          validation_count = validation_count + 1
        }
      }
      
      # Calculate accuracy as the ratio of correct predictions to the total 
      # number of samples
      training_accuracy <- training_count / length(Eval_Gray)
      validation_accuracy <- validation_count / length(Eval_Gray)
      epoch_training_accuracy <- signif(training_accuracy, 5)
      epoch_validation_accuracy <- signif(validation_accuracy, 5)
      epoch_training_loss <- signif(training_loss, 5)
      epoch_validation_loss <- signif(validation_loss, 5)
      
      pb$tick(tokens = list(training_accuracy = epoch_training_accuracy, 
                            validation_accuracy = epoch_validation_accuracy, 
                            training_loss = epoch_training_loss, 
                            validation_loss = epoch_validation_loss))
    }
    
    Stored_image_indexs <- sort(Stored_image_indexs)
    
    # Initialize an empty data frame to store the metrics
    Index_df <- data.frame(
      Index = Stored_image_indexs,
      learning_rate = current_lr
    )
    
    # Append the metrics to the data frame
    Gray_metrics_df <- rbind(Gray_metrics_df, data.frame(
      epoch = epoch,
      accuracy = epoch_training_accuracy,
      validation_accuracy = epoch_validation_accuracy,
      loss = epoch_training_loss,
      validation_loss = epoch_validation_loss
    ))
    
    if (epoch > 1)
    {
      # Create the ggplot graph for accuracy
      plot_accuracy <- ggplot(Gray_metrics_df, aes(x = epoch)) +
        geom_line(aes(y = accuracy, color = "Training Accuracy")) +
        geom_line(aes(y = validation_accuracy, color = "Validation Accuracy")) +
        geom_smooth(aes(y = accuracy, color = "Training Accuracy"),
                    method = "loess",
                    se = FALSE) +
        geom_smooth(
          aes(y = validation_accuracy, color = "Validation Accuracy"),
          method = "loess",
          se = FALSE
        ) +
        geom_point(
          aes(y = accuracy, color = "Training Accuracy"),
          shape = 1,
          size = 3 * 0.75
        ) +
        geom_point(
          aes(y = validation_accuracy, color = "Validation Accuracy"),
          shape = 2,
          size = 3 * 0.75
        ) +
        scale_color_manual(values = c(
          "Training Accuracy" = "lightblue",
          "Validation Accuracy" = "gold"
        )) +
        labs(x = NULL,
             y = "Accuracy",
             color = "Metrics") +
        theme_minimal() +
        theme(legend.position = "right") +
        guides(color = guide_legend(override.aes = list(shape = c(1, 2))))
      
      # Create the ggplot graph for loss
      plot_loss <- ggplot(Gray_metrics_df, aes(x = epoch)) +
        geom_line(aes(y = loss, color = "Training Loss")) +
        geom_line(aes(y = validation_loss, color = "Validation Loss")) +
        geom_smooth(aes(y = loss, color = "Training Loss"),
                    method = "loess",
                    se = FALSE) +
        geom_smooth(
          aes(y = validation_loss, color = "Validation Loss"),
          method = "loess",
          se = FALSE
        ) +
        geom_point(aes(y = loss, color = "Training Loss"),
                   shape = 1,
                   size = 3 * 0.75) +
        geom_point(
          aes(y = validation_loss, color = "Validation Loss"),
          shape = 2,
          size = 3 * 0.75
        ) +
        scale_color_manual(values = c(
          "Training Loss" = "lightblue",
          "Validation Loss" = "gold"
        )) +
        labs(x = "Epoch",
             y = "Loss",
             color = "Metrics") +
        theme_minimal() +
        theme(legend.position = "right") +
        guides(color = guide_legend(override.aes = list(shape = c(1, 2))))
      
      # Stack the accuracy and loss plots vertically
      stacked_plots <-
        grid.arrange(plot_accuracy, plot_loss, ncol = 1)
      
      # Display the stacked plots
      print(stacked_plots)
    }
    
    if (epoch < epochs)
    {
      # Set custom file names for saving the models
      lumos_gray_generator_model_name <-
        paste0("Lumos_gray_generator_", epoch, ".h5")
      lumos_gray_checkpoint_path <-
        file.path(Checkpoint_folder_location, lumos_gray_generator_model_name)
      
      # Save Gray_metrics_df and Index_df as CSV files
      write.csv(Gray_metrics_df, file.path(Checkpoint_folder_location, 
                                           "Gray_metrics.csv"), 
                row.names = FALSE)
      write.csv(Index_df, file.path(Checkpoint_folder_location, 
                                    "Index_df.csv"), row.names = FALSE)
      
      # Save the model
      save_model_hdf5(Lumos_gray_generator_model,
                      lumos_gray_checkpoint_path)
      
    } else
    {
      # Set custom file names for saving the models
      lumos_gray_generator_model_name <-
        paste0("Lumos_gray_generator.h5")
      lumos_gray_save_path <-
        file.path(Lumos_gray_path, 
                  lumos_gray_generator_model_name)
      
      # Save the model
      save_model_hdf5(Lumos_gray_generator_model,
                      lumos_gray_save_path)
    }
    
    # Clean up old checkpoint folders
    if (epoch > 1 && epoch != epochs) {
      previous_checkpoint_name <-
        paste0("Lumos_gray_generator_", epoch - 1, ".h5")
      previous_checkpoint_path <-
        file.path(Checkpoint_folder_location, previous_checkpoint_name)
      
      if (file.exists(previous_checkpoint_path)) {
        file.remove(previous_checkpoint_path)
      }
    } else if (epoch == epochs)
    {
      # After training, move the best models
      best_loss_model_checkpoint_path <- 
        file.path(Checkpoint_folder_location, 
                  "best_Lumos_Gray_loss_model.h5")
      best_accuracy_model_checkpoint_path <- 
        file.path(Checkpoint_folder_location, 
                  "best_Lumos_Gray_accuracy_model.h5")
      
      best_loss_model_final_path <- 
        file.path(Lumos_gray_generator_model_path, 
                  "best_Lumos_Gray_loss_model.h5")
      best_accuracy_model_final_path <- 
        file.path(Lumos_gray_generator_model_path, 
                  "best_Lumos_Gray_accuracy_model.h5")
      
      # Move the best loss model file
      file.rename(best_loss_model_checkpoint_path, 
                  best_loss_model_final_path)
      
      # Move the best accuracy model file
      file.rename(best_accuracy_model_checkpoint_path, 
                  best_accuracy_model_final_path)
      
      unlink(Checkpoint_folder_location, 
             recursive = TRUE,
             force = TRUE)
    }
    
    pb$terminate()
  }
})


###-------Build Lumos Binary model----------------------------------------------

Lumos_binary_environment <- new.env()

with(Lumos_binary_environment, {
  # Define Lumos binary model architecture
  Lumos_binary_generator_model <- keras_model_sequential() %>%
    layer_conv_2d(
      filters = 32,
      kernel_size = 3,
      activation = "relu",
      input_shape = c(image_size[1], image_size[2], 1)
    ) %>%
    layer_max_pooling_2d(pool_size = 2) %>%
    layer_conv_2d(filters = 64,
                  kernel_size = 3,
                  activation = "relu") %>%
    layer_max_pooling_2d(pool_size = 2) %>%
    layer_flatten() %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = prod(image_size), activation = "sigmoid") %>%
    layer_reshape(target_shape = c(image_size[1], image_size[2], 1))
  
  # Compile the model
  optimizer_binary <-
    tf$keras$optimizers$Adam(learning_rate = 0.0002)
  
  Lumos_binary_generator_model %>% compile(loss = "binary_crossentropy",
                                           optimizer = optimizer_binary, 
                                           metrics = list("accuracy"))
})

Lumos_binary_generator_model_path <-
  path.expand(Lumos_binary_path)

# Initialize an empty data frame to store the metrics
Binary_metrics_df <- data.frame(
  epoch = numeric(0),
  accuracy = numeric(0),
  validation_accuracy = numeric(0),
  loss = numeric(0),
  validation_loss = numeric(0)
)

with(Lumos_binary_environment, {
  # Create storage vaults for grayscale and binary images
  storage_vault_gray <- vector("list", length(total_samples))
  storage_vault_binary <- vector("list", length(total_samples))
  
  Stored_image_indexs <- c()
  
  # Training loop
  for (epoch in 1:epochs) {
    message("Epoch: ", epoch, "/", epochs)
    
    # Create the progress bar
    pb <- progress_bar$new(
      total = total_batches_needed,
      format = paste(
        ":current/:total [:bar] ETA: :eta - Accuracy: :training_accuracy - Loss: :training_loss - Validation accuracy: :validation_accuracy - Validation loss: :validation_loss" ,
        sep = ""
      ),
      clear = TRUE
    )
    
    training_accuracy <- 0
    training_loss <- 0
    validation_accuracy <- 0
    validation_loss <- 0
    
    # Create a checkpoint folder for saving the model
    Checkpoint_folder <- "Checkpoint_folder"
    Checkpoint_folder_location <-
      file.path(Lumos_binary_generator_model_path, Checkpoint_folder)
    
    if (epoch == 1) {
      dir.create(Checkpoint_folder_location)
    } 
    
    # Define callback for saving best model based on validation loss
    best_loss_model_checkpoint <- callback_model_checkpoint(
      filepath = paste0(Checkpoint_folder_location, 
                        "/best_Lumos_Binary_loss_model.h5"),
      monitor = "val_loss",
      save_best_only = TRUE,
      mode = "min"
    )
    
    # Define callback for saving best model based on validation accuracy
    best_accuracy_model_checkpoint <- callback_model_checkpoint(
      filepath = paste0(Checkpoint_folder_location, 
                        "/best_Lumos_Binary_accuracy_model.h5"),
      monitor = "val_accuracy",
      save_best_only = TRUE,
      mode = "max"
    )
    
    # Learning rate scheduler
    reduce_lr_callback <-
      callback_reduce_lr_on_plateau(
        monitor = "loss",
        factor = 0.1,
        patience = 2,
        verbose = 0,
        mode = "min",
        min_delta = 0.01,
        cooldown = 2,
        min_lr = 0.000025
      )
    
    # Create a callback to log metrics and save model checkpoints
    callbacks <- list(
      reduce_lr_callback,
      best_loss_model_checkpoint,
      best_accuracy_model_checkpoint
    )
    
    # Loop over the total number of batches
    for (i in 1:total_batches_needed) {
      
      # Create a random permutation of sample indices
      random_indices <- sample(total_samples, replace = FALSE)
      
      # Determine the start and end index for the current batch
      start_index <- 1
      end_index <- min(start_index + batch_size - 1, total_samples)
      
      # Get the random indices for the current batch
      batch_indices <- random_indices[start_index:end_index]
      
      # Create empty arrays to store batch data
      batch_images_gray <-
        array(
          dim = c(
            end_index - start_index + 1,
            gray_image_size[1],
            gray_image_size[2],
            gray_image_size[3]
          )
        )
      
      batch_images_binary <-
        array(
          dim = c(
            end_index - start_index + 1,
            binary_image_size[1],
            binary_image_size[2],
            binary_image_size[3]
          )
        )
      
      batch_labels <- array(dim = c(end_index - start_index + 1, 1))
      
      for (j in 1:length(batch_indices)) {
        index <- batch_indices[j]
        
        if (!(index %in% Stored_image_indexs)) {
          Stored_image_indexs <- append(Stored_image_indexs, index)
          
          Storage_images <-
            load_augmented_image(Train_Gray[index],
                                 Train_Binary[index],
                                 gray_image_size,
                                 binary_image_size,
                                 apply_augmentation = FALSE)
          
          # Store unaltered grayscale and binary images in the vaults
          storage_vault_gray[[index]] <- Storage_images$Primary
          storage_vault_binary[[index]] <- Storage_images$Secondary
        }
        
        Processed_images <-
          load_augmented_image(Train_Gray[index],
                               Train_Binary[index],
                               gray_image_size,
                               binary_image_size)
        
        # Add images and labels to the batch arrays
        batch_images_gray[j, , ,] <- Processed_images$Primary
        batch_images_binary[j, , ,] <- Processed_images$Secondary
        batch_labels[j,] <- index
      }
      
      # Generate a random index
      eval_random_index <- sample(total_samples, 1)
      
      # Get the random grayscale and binary image paths
      random_test_image <- Train_Gray[eval_random_index]
      random_train_image <- Train_Binary[eval_random_index]
      
      # Read the random grayscale and binary images
      eval_gray_image <- magick::image_read(random_test_image)
      eval_binary_image <- magick::image_read(random_train_image)
      
      # Convert magick images to arrays
      eval_gray_image_array <-
        as.numeric(magick::image_data(eval_gray_image))
      eval_binary_image_array <-
        as.numeric(magick::image_data(eval_binary_image))
      
      # Reshape image arrays to match the desired shape
      eval_reshaped_gray_image <-
        array_reshape(
          eval_gray_image_array,
          dim = c(
            1, 
            gray_image_size[1],
            gray_image_size[2],
            gray_image_size[3])
        )
      
      eval_reshaped_binary_image <-
        array_reshape(
          eval_binary_image_array,
          dim = c(
            1,
            binary_image_size[1],
            binary_image_size[2],
            binary_image_size[3]
          )
        )
      
      validation_images_gray <- eval_reshaped_gray_image
      validation_images_binary <- eval_reshaped_binary_image
      
      # Train the model
      history <- fit(
        Lumos_binary_generator_model,
        x = batch_images_gray,
        y = batch_images_binary,
        epochs = epochs,
        batch_size = batch_size,
        verbose = 0,
        validation_data = list(validation_images_gray, 
                               validation_images_binary),  
        callbacks = callbacks
      )
      
      # Acquire the learning rate in use
      current_lr <- history$metrics$lr[[length(history$metrics$lr)]]
      
      # Generate random indices to sample from storage vaults
      sample_index <- sample(Stored_image_indexs, 1)
      
      sampled_gray_images <- storage_vault_gray[[sample_index]]
      sampled_binary_images <- storage_vault_binary[[sample_index]]
      
      # Evaluate the generator model on training data
      training_evaluation <- 
        Lumos_binary_generator_model %>% evaluate(x = sampled_gray_images,
                                                  y = sampled_binary_images,
                                                  verbose = 0)
      
      # Evaluate the generator model on testing data
      validation_evaluation <-
        Lumos_binary_generator_model %>% evaluate(x = eval_reshaped_gray_image,
                                                  y = eval_reshaped_binary_image,
                                                  verbose = 0)
      
      # Retrieve the loss from the training evaluation
      training_loss <- training_evaluation["loss"]
      
      # Retrieve the loss from the validation evaluation
      validation_loss <- validation_evaluation["loss"]
      
      # Create empty lists to store images
      training_images_gray <- vector("list", length(Eval_Gray))
      training_images_binary <- vector("list", length(Eval_Gray))
      
      # Generate random indices to sample from storage vaults
      sample_indices <- sample(Stored_image_indexs, length(Eval_Gray))
      
      for (x in 1:length(sample_indices))
      {
        index <- sample_indices[x]
        
        sampled_gray_images <- storage_vault_gray[[index]]
        sampled_binary_images <- storage_vault_binary[[index]]
        
        training_images_gray[[x]] <- sampled_gray_images
        training_images_binary[[x]] <- sampled_binary_images
      }
      
      training_count = 0
      
      # Loop over the predicted and ground truth binary images
      for (i in 1:length(Eval_Binary)) {
        current_prediction <-
          Lumos_binary_generator_model %>% predict(x = training_images_gray[[i]], 
                                                   verbose = 0)
        
        # Calculate SSIM between the ground truth binary image and the 
        # predicted grayscale image
        ssim_score <-
          calculate_ssim(training_images_binary[[i]], current_prediction)
        
        ssim_value <- ssim_score$SSIM
        
        # Check if the SSIM score is above the similarity threshold (e.g., 90%)
        if (ssim_value >= target_training_accuracy)
        {
          training_count = training_count + 1
        }
      }
      
      # Create empty lists to store images
      eval_images_gray <- vector("list", length(Eval_Gray))
      eval_images_binary <- vector("list", length(Eval_Gray))
      
      # Loop over the samples in Eval_Gray
      for (j in 1:length(Eval_Gray)) {
        # Read the grayscale image
        gray_image <- magick::image_read(Eval_Gray[j])
        
        # Convert magick image to array
        gray_image_array <-
          as.numeric(magick::image_data(gray_image))
        
        # Reshape grayscale image array to match the desired shape
        reshaped_gray_image <- array_reshape(
          gray_image_array,
          dim = c(
            1,
            gray_image_size[1],
            gray_image_size[2],
            gray_image_size[3]
          )
        )
        
        # Add the reshaped grayscale image to the list
        eval_images_gray[[j]] <- reshaped_gray_image
        
        # Read the binary image
        binary_image <- magick::image_read(Eval_Binary[j])
        
        # Convert magick image to array
        binary_image_array <-
          as.numeric(magick::image_data(binary_image))
        
        # Reshape binary image array to match the desired shape
        reshaped_binary_image <- array_reshape(
          binary_image_array,
          dim = c(
            1,
            binary_image_size[1],
            binary_image_size[2],
            binary_image_size[3]
          )
        )
        
        # Add the reshaped binary image to the list
        eval_images_binary[[j]] <- reshaped_binary_image
      }
      
      validation_count = 0
      
      # Loop over the predicted and ground truth binary images
      for (i in 1:length(Eval_Binary)) {
        current_prediction <-
          Lumos_binary_generator_model %>% predict(x = eval_images_gray[[i]], 
                                                   verbose = 0)
        
        # Calculate SSIM between the ground truth grayscale image and the 
        # predicted binary image
        ssim_score <-
          calculate_ssim(eval_images_binary[[i]], current_prediction)
        
        ssim_value <- ssim_score$SSIM
        
        # Check if the SSIM score is above the similarity threshold (e.g., 90%)
        if (ssim_value >= target_validation_accuracy)
        {
          validation_count = validation_count + 1
        }
      }
      
      
      # Calculate accuracy as the ratio of correct predictions to the total 
      # number of samples
      training_accuracy <- training_count / length(Eval_Gray)
      validation_accuracy <- validation_count / length(Eval_Gray)
      epoch_training_accuracy <- signif(training_accuracy, 5)
      epoch_validation_accuracy <- signif(validation_accuracy, 5)
      epoch_training_loss <- signif(training_loss, 5)
      epoch_validation_loss <- signif(validation_loss, 5)
      
      pb$tick(tokens = list(training_accuracy = epoch_training_accuracy, 
                            validation_accuracy = epoch_validation_accuracy, 
                            training_loss = epoch_training_loss, 
                            validation_loss = epoch_validation_loss))
    }
    
    Stored_image_indexs <- sort(Stored_image_indexs)
    
    # Initialize an empty data frame to store the index and learning rate
    Index_df <- data.frame(
      Index = Stored_image_indexs,
      learning_rate = current_lr
    )
    
    # Append the metrics to the data frame
    Binary_metrics_df <- rbind(Binary_metrics_df, data.frame(
      epoch = epoch,
      accuracy = epoch_training_accuracy,
      validation_accuracy = epoch_validation_accuracy,
      loss = epoch_training_loss,
      validation_loss = epoch_validation_loss
    ))
    
    if (epoch > 1)
    {
      # Create the ggplot graph for accuracy
      plot_accuracy <- ggplot(Binary_metrics_df, aes(x = epoch)) +
        geom_line(aes(y = accuracy, color = "Training Accuracy")) +
        geom_line(aes(y = validation_accuracy, color = "Validation Accuracy")) +
        geom_smooth(aes(y = accuracy, color = "Training Accuracy"), 
                    method = "loess", 
                    se = FALSE) +
        geom_smooth(aes(y = validation_accuracy, color = "Validation Accuracy"), 
                    method = "loess", se = FALSE) +
        geom_point(aes(y = accuracy, color = "Training Accuracy"), 
                   shape = 1, size = 3 * 0.75) +  
        geom_point(aes(y = validation_accuracy, color = "Validation Accuracy"), 
                   shape = 2, size = 3 * 0.75) +  
        scale_color_manual(values = c("Training Accuracy" = "lightblue", 
                                      "Validation Accuracy" = "gold")) +
        labs(
          x = NULL,
          y = "Accuracy",
          color = "Metrics"
        ) +
        theme_minimal() +
        theme(legend.position = "right") +
        guides(color = guide_legend(override.aes = list(shape = c(1, 2))))
      
      # Create the ggplot graph for loss
      plot_loss <- ggplot(Binary_metrics_df, aes(x = epoch)) +
        geom_line(aes(y = loss, color = "Training Loss       ")) +
        geom_line(aes(y = validation_loss, color = "Validation Loss       ")) +
        geom_smooth(aes(y = loss, color = "Training Loss       "), 
                    method = "loess", 
                    se = FALSE) + 
        geom_smooth(aes(y = validation_loss, color = "Validation Loss       "), 
                    method = "loess", se = FALSE) +
        geom_point(aes(y = loss, color = "Training Loss       "), 
                   shape = 1, size = 3 * 0.75) +  
        geom_point(aes(y = validation_loss, color = "Validation Loss       "), 
                   shape = 2, size = 3 * 0.75) +  
        scale_color_manual(values = c("Training Loss       " = "lightblue", 
                                      "Validation Loss       " = "gold")) +
        labs(
          x = "Epoch",
          y = "Loss",
          color = "Metrics"
        ) +
        theme_minimal() +
        theme(legend.position = "right") +
        guides(color = guide_legend(override.aes = list(shape = c(1, 2))))
      
      # Stack the accuracy and loss plots vertically
      stacked_plots <- grid.arrange(plot_accuracy, plot_loss, ncol = 1)
      
      # Display the stacked plots
      print(stacked_plots)
    }
    
    if (epoch < epochs)
    {
      # Set custom file names for saving the models
      lumos_binary_generator_model_name <-
        paste0("Lumos_binary_generator_", epoch, ".h5")
      lumos_binary_checkpoint_path <-
        file.path(Checkpoint_folder_location, lumos_binary_generator_model_name)
      
      # Save Gray_metrics_df and Index_df as CSV files
      write.csv(Binary_metrics_df, file.path(Checkpoint_folder_location, 
                                             "Binary_metrics.csv"), 
                row.names = FALSE)
      write.csv(Index_df, file.path(Checkpoint_folder_location, 
                                    "Index_df.csv"), row.names = FALSE)
      
      # Save the model
      save_model_hdf5(Lumos_binary_generator_model,
                      lumos_binary_checkpoint_path)
      
    } else
    {
      # Set custom file names for saving the models
      lumos_binary_generator_model_name <-
        paste0("Lumos_binary_generator.h5")
      lumos_gray_save_path <-
        file.path(Lumos_binary_path, 
                  lumos_binary_generator_model_name)
      
      # Save the model
      save_model_hdf5(Lumos_binary_generator_model,
                      lumos_binary_save_path)
    }
    
    # Clean up old checkpoint folders
    if (epoch > 1 && epoch != epochs) {
      previous_checkpoint_name <-
        paste0("Lumos_binary_generator_", epoch - 1, ".h5")
      previous_checkpoint_path <-
        file.path(Checkpoint_folder_location, previous_checkpoint_name)
      
      if (file.exists(previous_checkpoint_path)) {
        file.remove(previous_checkpoint_path)
      }
    } else if (epoch == epochs)
    {
      # After training, move the best models
      best_loss_model_checkpoint_path <- 
        file.path(Checkpoint_folder_location, 
                  "best_Lumos_Binary_loss_model.h5")
      best_accuracy_model_checkpoint_path <- 
        file.path(Checkpoint_folder_location, 
                  "best_Lumos_Binary_accuracy_model.h5")
      
      best_loss_model_final_path <- 
        file.path(Lumos_binary_generator_model_path, 
                  "best_Lumos_Binary_loss_model.h5")
      best_accuracy_model_final_path <- 
        file.path(Lumos_binary_generator_model_path, 
                  "best_Lumos_Binary_accuracy_model.h5")
      
      # Move the best loss model file
      file.rename(best_loss_model_checkpoint_path, 
                  best_loss_model_final_path)
      
      # Move the best accuracy model file
      file.rename(best_accuracy_model_checkpoint_path, 
                  best_accuracy_model_final_path)
      
      unlink(Checkpoint_folder_location, 
             recursive = TRUE,
             force = TRUE)
    }
    
    pb$terminate()
  }
})


###-------Remove the temp folders-----------------------------------------------

Temporary_folder <- "Temporary_folder"
Train_Color_Path <- file.path(Train_Color_Path, Temporary_folder)
Train_Gray_Path <- file.path(Train_Gray_Path, Temporary_folder)
Color_Path <- file.path(Images_Path, Temporary_folder)


unlink(Train_Color_Path, recursive = TRUE)
unlink(Train_Gray_Path, recursive = TRUE)
unlink(Color_Path, recursive = TRUE)


###-------Indicate completion---------------------------------------------------
cat("Training of lumos models completed.")



