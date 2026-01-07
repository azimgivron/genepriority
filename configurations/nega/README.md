## NEGA Configuration File

#### Overview

The NEGA configuration file is a critical component of the **NEGA pipeline**. It specifies the parameters necessary for the evaluation command across various settings. This configuration file is designed to allow flexible data augmentation and optimization parameter adjustments for experiments using the original OMIM matrix.

For a detailed explanation of the NEGA framework, please refer to:  
*Ghaderi, S., Moreau, Y., & Ahookhosh, M. (2022). Non-Euclidean Gradient Methods: Convergence, Complexity, and Applications. Journal of Machine Learning Research, 23(2022):1-44.*

#### Data Configuration Settings

The pipeline supports multiple data configurations derived from the original OMIM matrix. Each setting modifies the data in different ways to enhance model training. The available configurations are:

1. **Original OMIM Matrix**  
   Contains only positive labels from the OMIM database.

2. **Original OMIM Matrix + Flip Label**  
   Implements a *flip label* approach where a subset of positive labels is sampled and re-assigned as negative labels for N epochs. This method introduces synthetic negative examples.

3. **Original OMIM Matrix + Randomly Sampled Negative Labels**  
   Enhances the dataset by appending negative labels that are randomly sampled, thus balancing the number of positive and negative examples.

4. **Original OMIM Matrix + Randomly Sampled Negative Labels + Flip Label**  
   Combines both random negative sampling and the flip label technique to further diversify and balance the dataset.

5. **Original OMIM Matrix + Side Information for Genes and Diseases**  
   Supplements the original matrix with additional contextual side information for both genes and diseases, which can provide richer feature representations.

6. **Original OMIM Matrix + Randomly Sampled Negative Labels + Side Information for Genes and Diseases**  
   Merges random negative sampling with side information, aiming for a more comprehensive enhancement of the input data.

7. **Original OMIM Matrix + Flip Label + Side Information for Genes and Diseases**  
   Integrates the flip label technique along with side information to balance the dataset while enriching it with auxiliary data.

8. **Original OMIM Matrix + Randomly Sampled Negative Labels + Flip Label + Side Information for Genes and Diseases**  
   Utilizes all available strategies by combining random negative sampling, flip label, and side information, offering the most enriched configuration.

#### Configuration Parameters

The following key parameters control the optimization process within the NEGA pipeline:

- **regularization_parameter (float)**  
  Regulates the strength of the regularization term to prevent overfitting.

- **symmetry_parameter (float)**  
  Adjusts the gradient update to ensure symmetric behavior in the optimization process.

- **lipschitz_smoothness (float)**  
  Sets the initial smoothness of the optimization landscape, influencing convergence.

- **rho_increase (float)**  
  Determines the factor by which the step size is increased dynamically during training to accelerate convergence under favorable conditions.

- **rho_decrease (float)**  
  Specifies the factor for decreasing the step size during optimization, helping to stabilize convergence as the algorithm refines the solution.

#### Usage Guidelines

1. **Choosing the Configuration:**  
   Select the data configuration that best aligns with your experimental needs. Consider whether you require synthetic negative labels, side information, or a combination of techniques to improve your model's performance.

2. **Tuning Parameters:**  
   Fine-tune the optimization parameters in the configuration file according to your dataset and experimental setup. Even minor adjustments in parameters like `regularization_parameter` or `rho_decrease` can have a notable impact on the optimization process.

3. **Running the Evaluation:**  
   Ensure that the configuration settings are consistent with your experimental design before executing the evaluation command. The NEGA pipeline reads these configurations at runtime to determine the appropriate processing steps.

4. **Extensibility:**  
   This configuration file is designed to be modular and extensible. Feel free to adapt and extend the file to include additional parameters or configurations as necessary for your research.

Below is a recap table where each column represents an augmentation, and a check (✔) or cross (✖) indicates whether that augmentation is applied in each setting:

| Setting ## | Original OMIM Matrix | Randomly Sampled Negative Labels | Flip Label | Side Information for Genes & Diseases |
|-----------|----------------------|----------------------------------|------------|---------------------------------------|
| 1         | ✔                    | ✖                                | ✖          | ✖                                     |
| 2         | ✔                    | ✖                                | ✔          | ✖                                     |
| 3         | ✔                    | ✔                                | ✖          | ✖                                     |
| 4         | ✔                    | ✔                                | ✔          | ✖                                     |
| 5         | ✔                    | ✖                                | ✖          | ✔                                     |
| 6         | ✔                    | ✔                                | ✖          | ✔                                     |
| 7         | ✔                    | ✖                                | ✔          | ✔                                     |
| 8         | ✔                    | ✔                                | ✔          | ✔                                     |

#### Conclusion

By providing multiple data augmentation options along with detailed optimization parameters, this configuration file offers a flexible framework to experiment with various approaches in the NEGA pipeline. Tailor these settings to your specific research goals to explore how different configurations affect model performance and generalization.
