# Gym Form Checker

The model will check, based on given video input, as to whether or not the current user has proper form for deadlifts and squats. A side profile is required in order to properly perform inference on.

## High Level Tasking:

1. Learn CV from small Kaggle intro. Or, from YT.
2. Setup small Pytorch library
3. Webscrape data from the sources below
4. Decide on:
   1. What to look for in video
      1. Barbell? Person's back? Weights?
      2. What model to use?
   2. What model would be appropriate
5. Format data as needed to input into model
6. Fix issues, TBD

## To-do list:

1. Project Scope:
   * Clearly define the problem you want to solve using machine learning.
     * The problem is that people continuously post as to whether or not their form, when performing lifts, is correct. Checking form during a powerlift, such as the deadlift, is very important, in order to prevent injury. The goal here is to ensure that user's can feed in their recorded lift and receive a confidence score for how good the form is.
   * Determine the project's goals, objectives, and desired outcomes.
     * Project must train both good form and bad form.
     * Project must be able to at least handle side profiles of users
     * Project must draw lines to indicate proper form for the following:
       * Straight back
       * Barbell direction
       * Final, straight leg position
       * Optional: Speed of lift
   * Identify the target audience or end-users for your project.
     * Users include but are not limited to:
       * Gym goers who wish to check lifting form
       * Coaches who use this to verify bad form
       * etc.
2. Gather and Prepare Data:
   * Determine the data requirements for your project and identify potential sources.

     * Good form:
       * Youtube, bodybuilding.com, or any workout website with strict, professional form completed by a coach or any other professional.
     * Bad form:
       * Posters on Reddit
       * Random user submissions
   * Collect the relevant data required for training and testing your models.

     * r/Fitness Daily Form Check Thread
     * r/formcheck
     * r/lifting

     ---
   * Preprocess and clean the data to remove noise, handle missing values, and normalize the features.
   * Split the data into training and testing sets.
3. Construct a Repository:
   * Set up a version control system (e.g., Git) to manage your project's codebase.
     * https://gitlab.com/spookypharaoh/gym-form-checker
   * Create a project directory structure to organize your code, data, and documentation.
   * Initialize a Git repository and commit your initial project files.
     * https://gitlab.com/spookypharaoh/gym-form-checker
4. Explore and Visualize the Data:
   * Perform exploratory data analysis to gain insights into the data.
   * Visualize the data using plots, histograms, and other visualization techniques.
   * Identify patterns, correlations, and outliers in the data.
5. Feature Engineering and Selection:
   * Engineer new features from existing ones that might improve model performance.
   * Select relevant features based on their importance and impact on the problem.
   * Apply dimensionality reduction techniques if needed (e.g., PCA, t-SNE).
6. Select and Train a Model:
   * Choose an appropriate machine learning algorithm based on your problem.
   * Split the training set into further subsets for training and validation.
   * Train the model on the training set and tune hyperparameters if necessary.
   * Validate the model's performance on the validation set.
7. Evaluate and Improve the Model:
   * Evaluate the model's performance on the testing set using appropriate metrics.
   * Analyze the model's strengths, weaknesses, and areas for improvement.
   * Iteratively refine the model by adjusting hyperparameters, trying different algorithms, or employing advanced techniques.
8. Deploy and Test the Model:
   * Prepare the model for deployment by packaging it with any necessary dependencies.
   * Create an interface or API to interact with the model.
   * Test the deployed model with new data or simulated user interactions.
   * Monitor the model's performance and collect feedback from users.
9. Document and Share the Project:
   * Document the entire project, including the problem statement, methodology, and findings.
   * Create a README file with instructions on how to use and replicate the project.
   * Share your project code, documentation, and findings through a GitHub repository or a personal website.
10. Maintain and Iterate:
    * Regularly update your project's codebase, especially if new data becomes available.
    * Stay updated with the latest advancements in machine learning techniques.
    * Continuously iterate and improve your models based on user feedback and new requirements.

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/spookypharaoh/gym-form-checker.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.com/spookypharaoh/gym-form-checker/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

---

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name

Choose a self-explaining name for your project.

## Description

Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges

On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals

Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation

Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage

Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support

Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap

If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing

State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment

Show your appreciation to those who have contributed to the project.

## License

For open source projects, say how it is licensed.

## Project status

If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
