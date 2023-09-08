

###-------User input------------------------------------------------------------

# Assign path for image input
path_in_images <- "~/path/to/images"

# Assign path for image output
Success_binary <- "~/path/for/successful/binary/images"
Success_gray <- "~/path/for/successful/grayscale/images"
Failed_grayscale <- "~/path/for/failed/grayscale/images"
Failed_binary <- "~/path/for/failed/binary/images"

# Assign model paths
Lumos_gray_generator_model_path <- "~/path/to/gray/models"
Lumos_binary_generator_model_path <- "~/path/to/binary/models"

# Assign a time limit (in seconds) for how long the code will try to alter images
time_limit_seconds <- 300 # in seconds

# Define target accuracy goals
similarity_threshold_gray <- 0.995 #e.g., 99.99% accuracy
similarity_thresold_binary <- 0.9995 #e.g., 99.995% accuracy

# Assign desired image size
image_size <- c(333, 250) # Width and height


###-------Load libraries--------------------------------------------------------

if (!require(tensorflow)) {
  install.packages("tensorflow")
  library(tensorflow)
} else
{
  library(tensorflow)
}

if (!require(progress)) {
  install.packages("progress")
  library(progress)
} else
{
  library(progress)
}

if (!require(magick)) {
  install.packages("magick")
  library(magick)
} else
{
  library(magick)
}

if (!require(keras)) {
  install.packages("keras")
  library(keras)
} else
{
  library(keras)
}

if (!require(SpatialPack)) {
  install.packages("SpatialPack")
  library(SpatialPack)
} else
{
  library(SpatialPack)
}

if (!require(BiocManager)) {
  install.packages("BiocManager")
  library(BiocManager)
} else 
{
  library(BiocManager)
}

if (!require(EBImage)) {
  install.packages("EBImage")
  library(EBImage)
} else
{
  library(EBImage)
}


###-------Create needed functions-----------------------------------------------

# Function to create the meta model by averaging predictions
# Arguments:
#   model1: The first Keras model to be included in the meta model
#   model2: The second Keras model to be included in the meta model
#   input_shape: The shape of the input data
# Returns:
#   A meta Keras model that combines the predictions of model1 and model2

create_meta_model <- function(model1, model2, input_shape) {
  
  # Create unique models
  unique_model1 <- keras_model_sequential(name = "unique_model1")
  unique_model1$add(model1)
  
  unique_model2 <- keras_model_sequential(name = "unique_model2")
  unique_model2$add(model2)
  
  # Define input layer
  input_layer <- layer_input(shape = input_shape)
  
  # Get outputs from unique models
  output1 <- unique_model1(input_layer)
  output2 <- unique_model2(input_layer)
  
  # Concatenate the output tensors from both unique models
  concatenated_output <- layer_concatenate(c(output1, output2))
  
  # Create and return the meta model
  meta_model <- keras_model(inputs = input_layer, outputs = concatenated_output)
  
  return(meta_model)
}



# Define image preprocessing function
# Arguments:
#   image_path: The path to the input image file
#   image_size: The target size (width and height) to which the image should be 
#   resized
# Returns:
#   A list containing the preprocessed image and padding information

preprocess_image <- function(image_path, image_size) {
  
  # Read the image
  image <- magick::image_read(image_path)
  
  # Get image dimensions
  image_info <- magick::image_info(image)
  img_width <- image_info$width
  img_height <- image_info$height
  
  # Initialize padding flags and values
  vertical_is_padded <- FALSE
  horizontal_is_padded <- FALSE
  pad_top <- pad_bottom <- pad_left <- pad_right <- 0
  
  # Check if vertical padding is needed due to image orientation
  if (img_height > img_width) {
    image <- magick::image_rotate(image, 90)
    img_width <- img_height
    img_height <- image_info$width
    vertical_is_padded <-
      TRUE  # Image was rotated, so it will be vertically padded
  }
  
  # Calculate aspect ratio
  aspect_ratio <- img_width / img_height
  
  # Check and apply padding based on aspect ratio
  if (aspect_ratio > image_size[1] / image_size[2]) {
    # Image is wider than the target aspect ratio, add vertical padding
    target_height <-
      round(image_size[2] * img_width / image_size[1])
    pad_top <- round((target_height - img_height) / 2)
    pad_bottom <- target_height - img_height - pad_top
    image <- magick::image_border(image, geometry = paste0("0x", pad_top,
                                                           "+0+", pad_bottom))
    vertical_is_padded <- TRUE  # Image was padded vertically
    
  } else {
    # Image is taller than the target aspect ratio, add horizontal padding
    target_width <-
      round(image_size[1] * img_height / image_size[2])
    pad_left <- round((target_width - img_width) / 2)
    pad_right <- target_width - img_width - pad_left
    image <-
      magick::image_border(image, geometry = paste0(pad_left, "x0+",
                                                    pad_right, "+0"))
    horizontal_is_padded <- TRUE  # Image was padded horizontally
  }
  
  # Convert to RGB color space and resize
  image <- magick::image_convert(image, colorspace = "RGB")
  image_resized <- magick::image_scale(image, image_size)
  
  # Calculate scaling factors
  scaling_factors <- list(
    width_scale_factor = image_size[1] / img_width,
    height_scale_factor = image_size[2] / img_height
  )
  
  # Store padding information
  padding_info <- list(
    horizontal_is_padded = horizontal_is_padded,
    vertical_is_padded = vertical_is_padded,
    pad_top = pad_top,
    pad_bottom = pad_bottom,
    pad_left = pad_left,
    pad_right = pad_right,
    scaling_factors = scaling_factors
  )
  
  return(list(image_resized, padding_info))
  
}



# Function to convert RGB image to grayscale with weights
# Arguments:
#   image: The RGB image to convert to grayscale
#   red_weight: Weight for the red channel in grayscale conversion
#   green_weight: Weight for the green channel in grayscale conversion
#   blue_weight: Weight for the blue channel in grayscale conversion
# Returns:
#   The grayscale image with weighted values

convert_to_grayscale <-
  function(image, red_weight, green_weight, blue_weight) {
    
    # Extract the color channels
    red_channel <- image[, , 1]
    green_channel <- image[, , 2]
    blue_channel <- image[, , 3]
    
    # Apply weights to the color channels
    weighted_red <- red_weight * red_channel
    weighted_green <- green_weight * green_channel
    weighted_blue <- blue_weight * blue_channel
    
    # Compute the grayscale values
    grayscale_values <- weighted_red + weighted_green + weighted_blue
    
    # Create a grayscale image with weighted values
    grayscale_image <- array(grayscale_values, dim = dim(image))
    
    # Return the grayscale image
    return(grayscale_image)
  }



# Calculate Structural Similarity Index (SSIM) between two images
# Arguments:
#   image1: The first image for SSIM comparison
#   image2: The second image for SSIM comparison
# Returns:
#   The SSIM score indicating the structural similarity between the two images

calculate_ssim <- function(image1, image2) {
  
  SSIM(image1, image2)
}



# Optimize a RGB weight based on a reward system
# Arguments:
#   weight: The weight to be optimized
#   attempted_weights: A vector of previously attempted weights
#   similarity_old: The old similarity score for comparison
#   similarity_new: The new similarity score for comparison
# Returns:
#   The optimized weight

optimize_weight <- function(weight, attempted_weights, similarity_old, 
                            similarity_new) {
  
  reward_factor <- 0.01 # Adjust this value based on desired sensitivity
  
  if (similarity_new >= similarity_old) {
    weight <- weight - reward_factor
  } else if (similarity_new < similarity_old) {
    weight <- weight + reward_factor
  }
  
  # Ensure the weight stays within valid bounds (e.g., [0, 1])
  weight <- pmin(1, pmax(0, weight))
  
  while (weight %in% attempted_weights) {
    
    # If the weight is attempted again, adjust it slightly
    weight <- weight + reward_factor * (-1)^sample(2, size = 1) 
    
    # Ensure the weight stays within valid bounds
    weight <- pmin(1, pmax(0, weight))
  }
  
  return(weight)
}



# Enhance an image with optional grayscale and binary processing
# Arguments:
#   image: The input image
#   file_extension: The file extension of the image
#   is_grayscale: Boolean, indicating if the image is grayscale
#   is_binary: Boolean, indicating if the image is binary
# Returns:
#   The enhanced image

enhanceimage <- function(image, file_extension, is_grayscale = FALSE, 
                         is_binary = FALSE) {
  
  if (is_grayscale || is_binary) {
    
    # Apply Gaussian blur for noise reduction using appropriate sigma
    
    if (is_grayscale) 
    {
      sigma <- 0.25 
    }
    else 
    {
      sigma <- 0.15
    }
    
    smoothed_image <- gblur(image, sigma = sigma)
    
    if (is_binary) {
      
      # Apply closing operation using structuring element
      modified_image <- closing(smoothed_image, 
                                kern = makeBrush(size = 1, 
                                                 shape = c("Gaussian")))
      
      enhanced_image <- modified_image
    } else {
      modified_image <- smoothed_image
      
      # Create a temporary file
      temp_file <- tempfile(pattern = "enhanced_", fileext = file_extension)
      temp_file <- paste0(temp_file, ".", file_extension)
      
      # Write the enhanced image to the temporary file
      writeImage(modified_image, temp_file, file_extension)
      
      # Read the content from the temporary file
      enhanced_image <- magick::image_read(temp_file)
      
      # Clean up by removing the temporary file
      file.remove(temp_file)
      
      # Apply further enhancement
      sharpened_image <- magick::image_contrast(enhanced_image, sharpen = 150)
      enhanced_image <- image_enhance(sharpened_image)
    }
  }
  
  return(enhanced_image)
}



# Calculate gradients for all color weights based on perturbation
# Arguments:
#   image: The input color image
#   model: The machine learning model for prediction
#   current_prediction: The current prediction of the grayscale image
#   gray_image: The grayscale version of the input image
#   red_weight: The weight for the red channel
#   green_weight: The weight for the green channel
#   blue_weight: The weight for the blue channel
#   similarity_threshold: The similarity threshold for gradient calculation
# Returns:
#   A list of gradients for each color channel (red, green, blue)

calculate_gradients <- function(image, model, current_prediction, gray_image, 
                                red_weight, green_weight, blue_weight,
                                similarity_threshold) {
  
  # Perturbation value for gradient calculation
  perturbation_value <- 0.001
  
  # Calculate gradients for each weight
  gradient_red <- calculate_color_gradient(image, model, current_prediction, 
                                           gray_image, 
                                           perturbation_value, red_weight, 
                                           green_weight, blue_weight, 
                                           similarity_threshold)
  gradient_green <- calculate_color_gradient(image, model, current_prediction, 
                                             gray_image, 
                                             perturbation_value, green_weight, 
                                             green_weight, blue_weight, 
                                             similarity_threshold)
  gradient_blue <- calculate_color_gradient(image, model, current_prediction, 
                                            gray_image, 
                                            perturbation_value, blue_weight, 
                                            green_weight, blue_weight, 
                                            similarity_threshold)
  
  # Return gradients as a list
  gradients <- list(gradient_red = gradient_red,
                    gradient_green = gradient_green,
                    gradient_blue = gradient_blue)
  
  return(gradients)
}



# Calculate the gradient for a single color weight
# Arguments:
#   image: The input color image
#   model: The machine learning model for prediction
#   current_prediction: The current prediction of the grayscale image
#   gray_image: The grayscale version of the input image
#   perturbation_value: The perturbation value for weight adjustment
#   weight: The current weight value for the color channel
#   green_weight: The weight for the green channel
#   blue_weight: The weight for the blue channel
#   similarity_threshold: The similarity threshold for gradient calculation
# Returns:
#   The gradient value for the specified color weight

calculate_color_gradient <- function(image, model, current_prediction, 
                                     gray_image, perturbation_value, weight, 
                                     green_weight, blue_weight, 
                                     similarity_threshold) {
  
  # Perturb the weight slightly
  perturbed_weight <- weight + perturbation_value
  
  # Calculate grayscale image with the perturbed weight
  perturbed_gray_image <- convert_to_grayscale(image, perturbed_weight, 
                                               green_weight, blue_weight)
  
  # Calculate SSIM with the perturbed grayscale image
  perturbed_prediction <- model %>% 
    predict(x = reshaped_color_image, verbose = 0)
  perturbed_ssim_score <- calculate_ssim(perturbed_prediction, 
                                         perturbed_gray_image)$SSIM
  
  # Calculate gradient based on the change in SSIM
  gradient <- (perturbed_ssim_score - similarity_new) / perturbation_value
  
  return(gradient)
}



# Optimize the binary threshold based on a reward system
# Arguments:
#   threshold: The current binary threshold value
#   attempted_thresholds: A list of previously attempted threshold values
#   similarity_old: The similarity score before threshold adjustment
#   similarity_new: The similarity score after threshold adjustment
#   reward_factor: The reward factor for adjusting the threshold
#   previous_direction: The previous adjustment direction (optional)
# Returns:
#   A list containing the optimized threshold and the updated previous direction

optimize_binary_threshold <- function(threshold, attempted_thresholds, 
                                      similarity_old, similarity_new, 
                                      reward_factor, previous_direction = NULL) {
  
  # Determine the direction of adjustment based on the improvement
  current_direction <- ifelse(similarity_new >= similarity_old, 1, -1)
  
  # If there is a previous direction and it's different from the current direction, update it
  if (!is.null(previous_direction) && current_direction != previous_direction) {
    previous_direction <- current_direction
  }
  
  # If there's no previous direction, use the current direction
  if (is.null(previous_direction)) {
    previous_direction <- current_direction
  }
  
  # Adjust the threshold based on the current direction and reward_factor
  threshold <- threshold + (previous_direction * reward_factor)
  
  # If the threshold has been attempted before, adjust it slightly
  while (threshold %in% attempted_thresholds) {
    threshold <- threshold + previous_direction * reward_factor * (-1)^sample(2, 
                                                                              size = 1)
  }
  
  return(list(threshold = threshold, previous_direction = previous_direction))
}


###-------Load and build models-------------------------------------------------

# Assign image sizes and channel counts for images 
color_image_size <-
  gray_image_size <- binary_image_size <- c(image_size[1],
                                            image_size[2],
                                            3)
gray_image_size[3] <- binary_image_size[3] <- 1

# Create paths to the models
Best_Lumos_gray_loss_path <-
  path.expand(paste0(Lumos_gray_generator_model_path, 
                     "/best_Lumos_Gray_loss_model.h5"))
Best_Lumos_gray_accuracy_path <-
  path.expand(paste0(Lumos_gray_generator_model_path,
                     "/best_Lumos_Gray_accuracy_model.h5"))
Best_Lumos_binary_loss_path <-
  path.expand(paste0(Lumos_binary_generator_model_path,
                     "/best_Lumos_Binary_loss_model.h5"))
Best_Lumos_binary_accuracy_path <-
  path.expand(paste0(Lumos_binary_generator_model_path,
                     "/best_Lumos_Binary_accuracy_model.h5"))

# Load grayscale models
Best_Lumos_gray_loss <-
  keras::keras$models$load_model(Best_Lumos_gray_loss_path)
Best_Lumos_gray_accuracy <-
  keras::keras$models$load_model(Best_Lumos_gray_accuracy_path)

# Load binary models
Best_Lumos_binary_loss <-
  keras::keras$models$load_model(Best_Lumos_binary_loss_path)
Best_Lumos_binary_accuracy <-
  keras::keras$models$load_model(Best_Lumos_binary_accuracy_path)

# Create meta model
meta_grayscale_model <- create_meta_model(Best_Lumos_gray_loss,
                                          Best_Lumos_gray_accuracy,
                                          color_image_size)
meta_binary_model <- create_meta_model(Best_Lumos_binary_loss,
                                       Best_Lumos_binary_accuracy,
                                       binary_image_size)

# Clear loaded models
Best_Lumos_gray_loss <- NULL
Best_Lumos_gray_accuracy <- NULL
Best_Lumos_binary_loss <- NULL
Best_Lumos_binary_accuracy <- NULL

# Compile the meta models
optimizer <-
  tf$keras$optimizers$Adam(learning_rate = 0.0002, beta_1 = 0.5)

meta_grayscale_model %>% compile(loss = "mean_squared_error",
                                 optimizer = optimizer)
meta_binary_model %>% compile(loss = "binary_crossentropy",
                              optimizer = optimizer)


###-------Preprocess images-----------------------------------------------------

# Load color images for processing
Color_images <-
  list.files(
    path_in_images, 
    pattern = ".jpg|.jpeg|.png", 
    full.names = TRUE,
    ignore.case = TRUE)

# Define paths and create a temporary folder
path_in_images <- path.expand(path_in_images)
Temporary_folder <- "Temporary_folder"
dir.create(file.path(path_in_images, Temporary_folder))
path_in_temporary <- file.path(path_in_images, Temporary_folder)

# Initialize lists and data frame
Resized_Color_images <- list()
Padded_df <- data.frame(
  Index = numeric(0),
  Vertical_is_padded = numeric(0),
  Horizontal_is_padded = numeric(0),
  Vertical_PadTop = numeric(0),
  Vertical_PadBottom = numeric(0),
  Horizontal_PadLeft = numeric(0),
  Horizontal_PadRight = numeric(0),
  Horizontal_scale = numeric(0),
  Vertical_scale = numeric(0)
)

pbar <- progress_bar$new(total = length(Color_images),
                         format = "[:bar] :percent ETA: :eta")

# Process images
for (i in 1:length(Color_images)) {
  
  # Preprocess the current image
  processed_image <- preprocess_image(Color_images[i], image_size)
  img <- processed_image[[1]]
  padding_info <- processed_image[[2]]
  
  # Extract padding information
  vertical_is_padded <- padding_info$vertical_is_padded
  horizontal_is_padded <- padding_info$horizontal_is_padded
  
  # Generate new file name
  file_name <- basename(Color_images[i])
  file_extension <- tools::file_ext(Color_images[i])
  new_file_name <- paste0(file_name, "_gray.", file_extension)
  
  # Check dimensions of the processed image
  img_width <- image_info(img)$width
  img_height <- image_info(img)$height
  if (img_width != 333 || img_height != 250) {
    message("Preprocessed image at index ",
            i, " has dimensions ", img_width, "x", img_height)
  }
  
  # Write the preprocessed image to the temporary folder
  magick::image_write(img, path = file.path(path_in_temporary, new_file_name))
  Resized_Color_images[[i]] <- img
  
  # Store padding information in the data frame
  if (vertical_is_padded || horizontal_is_padded) {
    Padded_df <- rbind(
      Padded_df,
      data.frame(
        Index = i,
        Vertical_is_padded = vertical_is_padded,
        Horizontal_is_padded = horizontal_is_padded,
        Vertical_PadTop = padding_info$pad_top,
        Vertical_PadBottom = padding_info$pad_bottom,
        Horizontal_PadLeft = padding_info$pad_left,
        Horizontal_PadRight = padding_info$pad_right,
        Horizontal_scale = padding_info$scaling_factors[[1]],
        Vertical_scale = padding_info$scaling_factors[[2]]
      ) 
    )
  } else 
  {
    Padded_df <- rbind(
      Padded_df,
      data.frame(
        Index = i,
        Vertical_is_padded = vertical_is_padded,
        Horizontal_is_padded = horizontal_is_padded,
        Vertical_PadTop = NA,
        Vertical_PadBottom = NA,
        Horizontal_PadLeft = NA,
        Horizontal_PadRight = NA,
        Horizontal_scale = NA,
        Vertical_scale = NA
      ) 
    )
  }
  
  pbar$tick()
}

pbar$terminate()

# List input images in the temporary folder
Input_images <-
  list.files(
    path_in_temporary,
    pattern = ".jpg|.jpeg|.png",
    full.names = TRUE,
    ignore.case = TRUE
  )
total_images = length(Input_images)


###-------Process images--------------------------------------------------------

# Initialize counters
counter <- 0
failed_counter_1 <- 0
failed_counter_2 <- 0

pbar <- progress_bar$new(total = total_images,
                         format = "[:bar] :percent ETA: :eta")

# Loop through images
for (i in 1:total_images)
{
  
  # Initialize timing
  start_time <- Sys.time()
  
  # Initialize variables
  points <- 0
  similarity_old <- 0
  Grayscale_flag = FALSE
  Binary_flag = FALSE
  attempted_red_weight <- list()
  attempted_green_weight <- list()
  attempted_blue_weight <- list()
  attempted_thresholds <- list()
  
  # Load the image
  Color_image_path <- Input_images[i]
  Color_image <- magick::image_read(Color_image_path)
  file_name <- basename(Color_image_path)
  file_extension <- tools::file_ext(Color_image_path)
  
  # Set intial color weights
  red_weight <- 0.2989
  green_weight <- 0.5870
  blue_weight <- 0.1140
  
  color_image_array <- as.numeric(magick::image_data(Color_image))
  reshaped_color_image <- array_reshape(
    color_image_array,
    dim = c(
      1, 
      color_image_size[1],
      color_image_size[2],
      color_image_size[3]))
  
  # Convert image to grayscale
  gray_image <- convert_to_grayscale(color_image_array, red_weight,
                                     green_weight, blue_weight)
  
  # Use Lumos gray to predict what the grayscale version would look like
  current_prediction <-
    meta_grayscale_model %>% predict(x = reshaped_color_image, verbose = 0)
  
  combined_grayscale_prediction <- ((current_prediction[, , , 1]) + 
                                      current_prediction[, , , 2]) / 2
  combined_grayscale_prediction <- array_reshape(combined_grayscale_prediction, 
                                                 dim = c(
                                                   1, 
                                                   gray_image_size[1],
                                                   gray_image_size[2],
                                                   gray_image_size[3]))
  
  # Calculate similarity between predicted and converted grayscale images
  ssim_score <- calculate_ssim(combined_grayscale_prediction, gray_image)
  
  similarity_value <- ssim_score$SSIM
  
  # Check if similarity is equal or above desired amount
  if (similarity_value >= similarity_threshold_gray)
  {
    Grayscale_flag = TRUE
  } else
  {
    similarity_new <- similarity_value
    
    # Continue alterations until image is acceptable or time runs out
    while (similarity_new < similarity_threshold_gray &&
           (difftime(Sys.time(), start_time, units = "secs") < 
            time_limit_seconds)) {
      
      # Add used weights to respective lists
      attempted_red_weight <- append(attempted_red_weight, red_weight)
      attempted_green_weight <- append(attempted_green_weight, green_weight)
      attempted_blue_weight <- append(attempted_blue_weight, blue_weight)
      
      # Calculate adjustment factor based on similarity
      adjustment_factor <- 1 / (1 + similarity_new)
      
      # Calculate gradients (partial derivatives) for each weight
      gradients <- calculate_gradients(color_image_array, 
                                       meta_grayscale_model, 
                                       combined_grayscale_prediction,
                                       red_weight, 
                                       green_weight, 
                                       blue_weight,
                                       similarity_threshold_gray) 
      
      gradient_red <- gradients$gradient_red
      gradient_green <- gradients$gradient_green
      gradient_blue <- gradients$gradient_blue
      
      # Adjust weights using gradients and adjustment factor
      red_weight <- red_weight + gradient_red * adjustment_factor
      green_weight <- green_weight + gradient_green * adjustment_factor
      blue_weight <- blue_weight + gradient_blue * adjustment_factor
      
      # Optimize red, green, and blue weights
      red_weight <- optimize_weight(red_weight, attempted_red_weight, 
                                    similarity_old, similarity_new)
      green_weight <- optimize_weight(green_weight, attempted_green_weight, 
                                      similarity_old, similarity_new)
      blue_weight <- optimize_weight(blue_weight, attempted_blue_weight, 
                                     similarity_old, similarity_new)
      
      # Update similarity values
      similarity_old <- similarity_value
      
      # Generate new grayscale image
      gray_image <- convert_to_grayscale(color_image_array,
                                         red_weight,
                                         green_weight,
                                         blue_weight)
      
      # Calculate similarity between predicted and converted grayscale image
      ssim_score <- calculate_ssim(combined_grayscale_prediction, gray_image)
      similarity_value <- ssim_score$SSIM
      
      similarity_new <- similarity_value
    }
    
    # Check if time limit was exceeded
    if (difftime(Sys.time(), start_time, units = "secs") > 
        time_limit_seconds) {
      failed_counter_1 = failed_counter_1 + 1
    } else {
      Grayscale_flag = TRUE
    }
  }
  
  if (Grayscale_flag == TRUE)
  {
    
    # Reset the similarity value for the new image conversion attempt
    similarity_old <- 0
    
    # Combine layers of grayscale images
    grayscale_image <- RGB2gray(gray_image)
    
    Gray_image_altered <- enhanceimage(grayscale_image, file_extension, 
                                       is_grayscale = TRUE)
    
    Gray_image_array <-
      as.numeric(magick::image_data(Gray_image_altered))
    
    reshaped_gray_image <- array_reshape(
      Gray_image_array,
      dim = c(
        gray_image_size[1],
        gray_image_size[2],
        gray_image_size[3]))
    
    # Extract the name of the grayscale image without the "_gray" suffix
    grayscale_image_name <-
      gsub(paste0("(_gray)?\\.", file_extension), "",
           file_name)
    
    # Define the path to save the converted grayscale image
    grayscale_image_path <- file.path(Success_gray,
                                      paste0(grayscale_image_name, "_Gray.",
                                             file_extension))
    
    # Write the grayscale image to the specified path
    writeImage(reshaped_gray_image, grayscale_image_path)
    
    # List the converted grayscale images in the Success_gray directory
    converted_grayscale_images <- list.files(
      Success_gray,
      pattern = ".jpg|.jpeg|.png",
      full.names = TRUE,
      ignore.case = TRUE
    )
    
    # Convert the grayscale image data to a numeric array
    Gray_image_array <- as.numeric(magick::image_data(Gray_image_altered))
    
    # Define initial threshold, window size, and reward factor
    threshold <- 0.5
    reward_factor <- 0.01
    iteration_counter <- 0
    best_threshold <- threshold
    best_similarity <- 0
    num_windows <- 10
    
    binary_image_adjusted <- ifelse(Gray_image_array > threshold, 1, 0)
    
    # Reshape the grayscale image array to match the binary image size
    reshaped_gray_image <- array_reshape(
      binary_image_adjusted,
      dim = c(
        1,
        binary_image_size[1],
        binary_image_size[2],
        binary_image_size[3]
      )
    )
    
    # Use Lumos binary to predict what the binary version would look like
    current_prediction <- meta_binary_model %>%
      predict(x = reshaped_gray_image, verbose = 0)
    
    combined_binary_prediction <- ((current_prediction[, , , 1]) + 
                                     current_prediction[, , , 2]) / 2
    
    combined_binary_prediction <- ifelse(Gray_image_array > 0.6, 1, 0)
    
    combined_binary_prediction <- array_reshape(combined_binary_prediction, 
                                                dim = c(
                                                  1,
                                                  binary_image_size[1],
                                                  binary_image_size[2],
                                                  binary_image_size[3]))
    
    # Calculate similarity between predicted and converted grayscale images
    ssim_score <-
      calculate_ssim(combined_binary_prediction, reshaped_gray_image)
    similarity_value <- ssim_score$SSIM
    
    best_similarity <- similarity_value
    
    if (similarity_value >= similarity_thresold_binary)
    {
      Binary_flag = TRUE
    }
    else
    {
      # Initialize similarity_new for comparison
      similarity_old <- similarity_value
      
      # Set the threshold to detect if 100% is white
      white_pixel_threshold <- binary_image_size[1] * binary_image_size[2]
      
      Gray_image_altered <- enhanceimage(Gray_image_array, file_extension, 
                                         is_grayscale = FALSE, is_binary = TRUE)
      
      # Reshape the morph_cleaned_image to the desired dimensions
      morph_image_array <- array_reshape(Gray_image_altered,
                                         dim = c(
                                           1, 
                                           gray_image_size[1], 
                                           gray_image_size[2], 
                                           gray_image_size[3]))
      
      benchmark_dfs <- data.frame(
        threshold = numeric(0),
        similarity_new = numeric(0),
        similarity_old = numeric(0)
      )
      
      # Perform a benchmark test
      benchmark_thresholds <- seq(from = 0.1, to = 0.9, by = 0.1)
      
      for (benchmark in 1:length(benchmark_thresholds))
      {
        # Acquire current threshold
        testing_threshold <- benchmark_thresholds[benchmark]
        
        testing_image_adjusted <- ifelse(morph_image_array > 
                                           testing_threshold, 1, 0)
        
        # Calculate similarity between predicted and thresholded images
        ssim_score <-
          calculate_ssim(combined_binary_prediction, testing_image_adjusted)
        similarity_value <- ssim_score$SSIM
        
        # Create a new row to add to the data frame
        new_row <- data.frame(
          threshold = testing_threshold,
          similarity_new = similarity_value,
          similarity_old = similarity_old
        )
        
        # Append the new row to the similarity data frame
        benchmark_dfs <- rbind(benchmark_dfs, new_row)
      }
      
      # Create an empty list to store the data frames
      similarity_dfs <- list()
      
      # Loop to create and update the data frames
      for (dataframe in 1:10) {
        variable_name <- paste("similarity_df_", dataframe , sep = "")
        variable_contents <- data.frame(
          threshold = numeric(0),
          similarity_new = numeric(0),
          similarity_old = numeric(0)
        )
        
        # Assign the data frame to the list
        similarity_dfs[[dataframe]] <- variable_contents
        
        # Update the list with the modified data frame
        similarity_dfs[[dataframe]] <- benchmark_dfs
      }
      
      # Find the index of the row with the highest similarity_new
      max_index <- which.max(similarity_dfs[[1]]$similarity_new)
      
      # Retrieve the threshold with the highest similarity_new
      best_threshold <- similarity_dfs[[1]]$threshold[max_index]
      best_similarity <- similarity_new <- 
        similarity_dfs[[1]]$similarity_new[max_index]
      similarity_old <- similarity_dfs[[1]]$similarity_old[max_index]
      
      binary_image_adjusted <- ifelse(Gray_image_array > 
                                        best_threshold, 1, 0)
      
      # Continue adjusting the threshold until reaching the similarity threshold 
      # or time limit
      while (similarity_new < similarity_thresold_binary &&
             difftime(Sys.time(), start_time, units = "secs") < 
             time_limit_seconds) {
        cat("\n Number:", 392)
        # Set local threshold
        increase_factor <- 0.5
        Local_threshold <- similarity_thresold_binary^increase_factor
        
        # Calculate window dimensions based on desired number of windows
        window_width <- floor(binary_image_size[1] / num_windows)
        window_height <- floor(binary_image_size[2] / num_windows)
        
        # Create empty matrix to store thresholded windows
        thresholded_windows <- matrix(0, nrow = binary_image_size[1], 
                                      ncol = binary_image_size[2])
        
        windows <- expand.grid(horizontal_window = seq(1, 5, 1), 
                               vertical_window = seq(1, 2, 1))
        
        Index <- seq(1:10)
        
        windows['Index'] <- Index
        
        testing_thresholds <- list()
        
        previous_direction <- NULL
        
        for (m in 1:2) {
          for (k in 1:5) {
            j <- windows$Index[windows$horizontal_window == k &
                                 windows$vertical_window == m]
            
            # Define row and column ranges for the current window
            row_range <- ((m - 1) * window_height + 1):(m * window_height)
            col_range <- ((k - 1) * window_width + 1):(k * window_width)
            
            # Extract window from grayscale image and prediction image
            local_gray_window <- morph_image_array[, row_range, col_range, ]
            local_prediction_window <- 
              combined_binary_prediction[, row_range, col_range, ]
            
            # Find the index of the row with the highest similarity_new
            max_index <- which.max(similarity_dfs[[j]]$similarity_new)
            
            # Retrieve the threshold with the highest similarity_new
            threshold <- similarity_dfs[[j]]$threshold[max_index]
            best_similarity <- similarity_new <- 
              similarity_dfs[[j]]$similarity_new[max_index]
            similarity_old <- similarity_dfs[[j]]$similarity_old[max_index]
            
            if (j > 1)
            {
              # Find the index of the row with the highest similarity_new
              prev_max_index <- which.max(similarity_dfs[[j-1]]$similarity_new)
              prev_best_threshold <- 
                similarity_dfs[[j-1]]$threshold[prev_max_index]
            } else
            {
              prev_best_threshold <- NULL
            }
            
            window_df <- similarity_dfs[[j]]
            
            while (similarity_new < Local_threshold &&
                   difftime(Sys.time(), start_time, units = "secs") < 
                   time_limit_seconds) {
              
              attempted_thresholds <- similarity_dfs[[j]]$threshold
              
              if (!is.null(prev_best_threshold))
              {
                
                # Apply threshold to the window
                window_thresholded_image <- ifelse(local_gray_window > 
                                                     threshold, 1, 0)

                # Calculate similarity between predicted and thresholded images
                ssim_score <- calculate_ssim(local_prediction_window, 
                                             window_thresholded_image)
                window_similarity_value <- ssim_score$SSIM

                # Apply threshold to the window
                prev_threshold_thresholded_image <- ifelse(local_gray_window > 
                                                           prev_best_threshold, 
                                                           1, 0)
                
                # Calculate similarity between predicted and thresholded images
                ssim_score <- calculate_ssim(local_prediction_window, 
                                             prev_threshold_thresholded_image)
                prev_window_similarity_value <- ssim_score$SSIM
                
                if (prev_window_similarity_value > window_similarity_value)
                {
                  threshold <-prev_best_threshold
                  
                  thresholded_image <- prev_threshold_thresholded_image
                  
                  # Calculate similarity between predicted and thresholded images
                  ssim_score <- calculate_ssim(local_prediction_window, 
                                               thresholded_image)
                  similarity_value <- ssim_score$SSIM
                } else
                {
                  threshold <- threshold
                  
                  thresholded_image <- window_thresholded_image
                  
                  # Calculate similarity between predicted and thresholded images
                  ssim_score <- calculate_ssim(local_prediction_window, 
                                               thresholded_image)
                  similarity_value <- ssim_score$SSIM
                }
                prev_best_threshold <- NULL  
              } else
              {
                # Optimize the binary threshold using the reward system
                optimization_result <- 
                  optimize_binary_threshold(threshold, attempted_thresholds, 
                                            similarity_old, similarity_new, 
                                            reward_factor, previous_direction)
                
                threshold <- optimization_result$threshold
                
                previous_direction <- optimization_result$previous_direction
                
                # Apply threshold to the window
                thresholded_image <- ifelse(local_gray_window > 
                                              threshold, 1, 0)

                # Calculate similarity between predicted and thresholded images
                ssim_score <- calculate_ssim(local_prediction_window, 
                                             thresholded_image)
                similarity_value <- ssim_score$SSIM
                
              }
 
              # Update the similarity and attempted thresholds for the current 
              # window
              similarity_old <- similarity_new
              similarity_new <- similarity_value
              
              # Create a new row to add to the data frame
              new_row <- data.frame(
                threshold = threshold,
                similarity_new = similarity_new,
                similarity_old = similarity_old
              )
              
              similarity_dfs[[j]] <- rbind(similarity_dfs[[j]], new_row)
              
              window_df <- 
                rbind(window_df, new_row)
              
              # Update the best threshold and similarity if applicable
              if (similarity_new > best_similarity) {
                best_similarity <- similarity_new
                best_threshold <- threshold
              }
              
              # Count the number of white pixels in the binary image
              num_white_pixels <- sum(thresholded_windows == 1)
              
              if (num_white_pixels == white_pixel_threshold) {
                # If the image is predominantly white, reduce the threshold
                threshold <- threshold - reward_factor * 0.1
              }
              
              # Increment the iteration counter
              iteration_counter <- iteration_counter + 1
              
              # Periodic adjustment every 250 iterations
              if (iteration_counter %% 100 == 0 || threshold > 1 ||
                  threshold < 0) 
              {
                threshold <- best_threshold
                reward_factor <- reward_factor * 0.9  # Diminishing reward factor
                iteration_counter <- 0
              }
              
            }
            testing_thresholds <- c(testing_thresholds, best_threshold)
            
            # Recombine the thresholded windows
            thresholded_windows[row_range, col_range] <- thresholded_image
          }
        }
        
        best_similarity <- 0
        
        thresholded_image_reshaped <- array_reshape(
          thresholded_windows, 
          dim = c(
            1,
            binary_image_size[1],
            binary_image_size[2],
            binary_image_size[3]
          )
        )
        
        # Calculate similarity between predicted and converted grayscale images
        ssim_score <-
          calculate_ssim(combined_binary_prediction, thresholded_image_reshaped)
        similarity_value <- ssim_score$SSIM 
        
        # Update the best threshold and similarity if applicable
        if (similarity_new > best_similarity) {
          best_similarity <- similarity_new
        }
        
        if (similarity_value > similarity_thresold_binary)
        {
          similarity_old <- similarity_new
          similarity_new <- similarity_value
          final_image <- thresholded_image_reshaped
        } else 
        {
          for (test in 1:length(testing_thresholds))
          {
            threshold <- testing_thresholds[[test]]
            
            binary_image_adjusted <- ifelse(morph_image_array > threshold, 1, 0)
            
            # Calculate similarity between predicted and converted grayscale 
            # images
            ssim_score <-
              calculate_ssim(combined_binary_prediction, binary_image_adjusted)
            similarity_value <- ssim_score$SSIM
            
            if (similarity_value > best_similarity)
            {
              final_image <- binary_image_adjusted
              similarity_old <- similarity_new
              similarity_new <- similarity_value
              best_threshold <- threshold
            }
          }
        }
        
        if (j == 10 && similarity_value > similarity_thresold_binary)
        {
          binary_image_adjusted <- final_image
        }

      }
      
      # Check if time limit was exceeded
      if (similarity_new < similarity_thresold_binary && 
          (difftime(Sys.time(), start_time, units = "secs") > 
           time_limit_seconds)) {
        
        failed_counter_2 <- failed_counter_2 + 1
      } else {
        Binary_flag = TRUE
      }
    }
  }
  else
  {
    # Define the directory name for failed grayscale images
    renamed_dir <- "Failed_Grayscale"
    
    # Create a new file name by appending "_Failed_Grayscale" to the original 
    # name
    renamed_dir <- paste0(file_name, "_", renamed_dir, ".", file_extension)
    
    # Modify binary_image_name to add "_Binary" suffix
    grayscale_image_name <- gsub(paste0("(_gray)?\\.", file_extension), "", 
                                 file_name)
    grayscale_image_name <- paste0(grayscale_image_name, "_Gray", ".", 
                                   file_extension)
    
    # Create a file path to save the image
    failed_image_path <- file.path(paste0(Failed_grayscale, "/", 
                                          grayscale_image_name))
    
    # Save the image
    writeImage(gray_image, failed_image_path)
  }
  
  if (Binary_flag == TRUE )
  {
    
    # Reshape the binary image
    binary_image <- Image(array_reshape(
      binary_image_adjusted,
      dim = c(
        binary_image_size[1],
        binary_image_size[2])))
    
    # Create a temporary directory to store the binary image
    temp_file <-
      (file.path(paste0(Success_binary, "/Temporary_file")))
    dir.create(temp_file)
    
    # Define the temporary image path and save the binary image
    temporary_image_path <- file.path(temp_file,
                                      paste0(file_name, "_Temporary.",
                                             file_extension))
    writeImage(binary_image, temporary_image_path)
    
    # List the temporary files in the directory
    binary_read <- list.files(
      temp_file,
      pattern = ".jpg|.jpeg|.png",
      full.names = TRUE,
      ignore.case = TRUE
    )
    
    # Read the binary image
    binary_image <- magick::image_read(binary_read)
    
    # Remove the temporary directory and its contents
    unlink(temp_file, recursive = TRUE)
    
    # Retrieve padding information
    vertical_pad <- Padded_df$Vertical_is_padded[Padded_df$Index == i]
    horizontal_pad <- Padded_df$Horizontal_is_padded[Padded_df$Index == i]
    
    if (vertical_pad || horizontal_pad) {
      
      # Retrieve padding values
      vertical_pad_top <- Padded_df$Vertical_PadTop[Padded_df$Index == i]
      vertical_pad_bottom <-
        Padded_df$Vertical_PadBottom[Padded_df$Index == i]
      horizontal_pad_left <-
        Padded_df$Horizontal_PadLeft[Padded_df$Index == i]
      horizontal_pad_right <-
        Padded_df$Horizontal_PadRight[Padded_df$Index == i]
      
      # Retrieve scaling factors
      Horizontal_scale <-
        Padded_df$Horizontal_scale[Padded_df$Index == i]
      Vertical_scale <-
        Padded_df$Vertical_scale[Padded_df$Index == i]
      
      # Calculate cropping dimensions
      crop_width <- horizontal_pad_left * Horizontal_scale
      crop_height <- vertical_pad_top * Vertical_scale
      
      # Crop the binary image to remove padding from the left and bottom
      binary_image <- magick::image_crop(binary_image,
                                         geometry = paste0("-",
                                                           crop_width,
                                                           "-",
                                                           crop_height))
      
      # Crop the binary image to remove padding from the right and top
      binary_image <- magick::image_crop(binary_image,
                                         geometry = paste0("+",
                                                           crop_width,
                                                           "+",
                                                           crop_height))
    }
    
    # Define the path and name for the binary image
    binary_image_path <-
      file.path(Success_binary,
                paste0(file_name, "_Binary.",
                       file_extension))
    binary_image_name <-
      gsub(paste0("(_gray)?\\.", 
                  file_extension), "", file_name)
    binary_image_path <-
      file.path(Success_binary,
                paste0(binary_image_name, "_Binary.", file_extension))
    
    # Write the binary image to the specified path
    image_write(binary_image, path = binary_image_path, format = file_extension)
    
    # Increment the counter for successful binary images
    counter = counter + 1
  } else if (Binary_flag == FALSE && Grayscale_flag == TRUE)
  {
    # Define the directory name for failed binary images
    renamed_dir <- "Failed_Binary"
    
    # Create a new file name by appending "_Failed_Binary" to the original 
    # name
    renamed_dir <- paste0(file_name, "_", renamed_dir, ".", file_extension)
    
    # Modify binary_image_name to add "_Binary" suffix
    binary_image_name <- gsub(paste0("(_gray)?\\.", file_extension), "", 
                              file_name)
    binary_image_name <- paste0(binary_image_name, "_Binary", ".", 
                                file_extension)
    
    # Create a file path to save the image
    failed_image_path <- file.path(paste0(Failed_binary, "/", 
                                          binary_image_name))
    
    # Reshape the binary image
    failed_binary_image <- Image(array_reshape(
      binary_image_adjusted,
      dim = c(
        binary_image_size[1],
        binary_image_size[2])))
    
    # Save the image
    writeImage(failed_binary_image, failed_image_path)
  }
  
  # Get a list of grayscale images to remove
  Remove_Gray <-
    list.files(Success_gray, pattern = ".jpg|.jpeg|.png",
               full.names = TRUE)
  file.remove(Remove_Gray)
  
  pbar$tick()
}

pbar$terminate()


###-------Indicate completion---------------------------------------------------
cat("Image conversion completed \n")
cat(counter,
    "images successfully converted from color images to binary images. \n")
cat(
  failed_counter_1,
  "images failed while converting from color images to grayscale images. \n")
cat(
  failed_counter_2,
  "images failed while converting from grayscale images to binary images. \n")
