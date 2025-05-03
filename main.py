import os
import json
import google.generativeai as genai
import google.ai.generativelanguage as glm
from github import Github
from github.GithubException import UnknownObjectException, GithubException
from flask import Flask, request, jsonify
import logging # Optional, but good for seeing logs on Render

# --- Flask App Setup ---
app = Flask(__name__)

# --- Optional Logging Setup ---
logging.basicConfig(level=logging.INFO) # Log messages to Render's logs

# ==============================================================================
# === 1. CORE FUNCTION: GitHub Update (Same as before) ========================
# ==============================================================================
def update_github_file(token: str, repo_name: str, file_path: str, new_content: str, commit_message: str, branch: str = "main"):
    """
    Updates a file in a specified GitHub repository using a Personal Access Token (PAT).
    Returns a dict: {'status': 'success'|'error', 'message': '...'}
    """
    app.logger.info(f"Attempting GitHub update: {repo_name}/{file_path} on branch {branch}")
    try:
        g = Github(token)
        try:
            repo = g.get_repo(repo_name)
            app.logger.info(f"Accessed repo: {repo.full_name}")
        except UnknownObjectException:
            msg = f"Repository '{repo_name}' not found or token lacks permissions."
            app.logger.error(msg)
            return {"status": "error", "message": msg}
        except GithubException as e:
             msg = f"Error accessing repository '{repo_name}': Status {e.status}"
             app.logger.error(msg)
             return {"status": "error", "message": msg}

        try:
            contents = repo.get_contents(file_path, ref=branch)
            current_sha = contents.sha
            app.logger.info(f"Found existing file '{file_path}' (SHA: {current_sha})")
        except UnknownObjectException:
             msg = f"File '{file_path}' not found in branch '{branch}'. Cannot update non-existent file."
             app.logger.error(msg)
             return {"status": "error", "message": msg}
        except GithubException as e:
             error_code = f"GitHub API error getting file details for '{file_path}': Status {e.status}"
             if e.status == 404: msg = f"File '{file_path}' not found (returned 404)."
             elif e.status == 403: msg = "Forbidden accessing file. Check PAT permissions ('repo' scope) or rate limiting."
             else: msg = error_code
             app.logger.error(f"{error_code} - {msg}")
             return {"status": "error", "message": msg}

        app.logger.info(f"Attempting to update '{file_path}'...")
        commit_info = repo.update_file(
            path=file_path, message=commit_message, content=new_content,
            sha=current_sha, branch=branch
        )
        commit_sha = commit_info['commit'].sha
        success_msg = f"File '{file_path}' updated successfully in branch '{branch}'. Commit SHA: {commit_sha}"
        app.logger.info(f"Success: {success_msg}")
        return {"status": "success", "message": success_msg}

    except GithubException as e:
        error_code = f"GitHub API error during update: Status {e.status}"
        if e.status == 401: msg = "GitHub Authentication failed. Check PAT validity/permissions."
        elif e.status == 403: msg = "GitHub Forbidden during update. Check PAT permissions/rate limits."
        elif e.status == 404: msg = "GitHub Repo/file not found during update attempt."
        elif e.status == 409: msg = "GitHub Conflict detected updating file."
        else: msg = f"{error_code}, Data: {e.data}"
        app.logger.error(f"{error_code} - {msg}")
        return {"status": "error", "message": msg}
    except Exception as e:
        error_msg = f"Unexpected error during GitHub update: {e}"
        app.logger.exception(error_msg) # Log traceback for unexpected errors
        return {"status": "error", "message": error_msg}

# ==============================================================================
# === 2. GEMINI TOOL SCHEMA (Same as before) ===================================
# ==============================================================================
update_github_file_func_declaration = glm.FunctionDeclaration(
    name="update_github_file",
    description="Updates a specified file in a given GitHub repository...", # Keep full description
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        properties={
            "repo_name": glm.Schema(type=glm.Type.STRING, description="..."),
            "file_path": glm.Schema(type=glm.Type.STRING, description="..."),
            "new_content": glm.Schema(type=glm.Type.STRING, description="..."),
            "commit_message": glm.Schema(type=glm.Type.STRING, description="..."),
            "branch": glm.Schema(type=glm.Type.STRING, description="...")
        },
        required=["repo_name", "file_path", "new_content", "commit_message"]
    )
)
gemini_tools = [glm.Tool(function_declarations=[update_github_file_func_declaration])]
available_functions = { "update_github_file": update_github_file }

# ==============================================================================
# === 3. FLASK WEB ENDPOINT ====================================================
# ==============================================================================
@app.route('/update', methods=['POST'])
def handle_update_request():
    """Handles POST requests to update a GitHub file via Gemini."""
    app.logger.info("Received request on /update endpoint.")

    # --- Get Input ---
    if not request.is_json:
        app.logger.warning("Request is not JSON.")
        return jsonify({"status": "error", "message": "Request must be JSON."}), 400

    data = request.get_json()
    user_prompt = data.get('prompt')

    if not user_prompt:
        app.logger.warning("Missing 'prompt' in JSON payload.")
        return jsonify({"status": "error", "message": "Missing 'prompt' in JSON payload."}), 400
    app.logger.info(f"Received prompt: {user_prompt[:100]}...") # Log beginning of prompt

    # --- Configuration & Initialization (inside request for Render env vars) ---
    try:
        gemini_api_key = os.environ.get("GOOGLE_API_KEY")
        github_token = os.environ.get("GITHUB_TOKEN") # Load GitHub token here

        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        if not github_token:
             # Don't expose this specific error to the client, log it server-side
            app.logger.error("GITHUB_TOKEN environment variable not set on server.")
            raise ValueError("Server configuration error.") # Raise generic error for client

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest", # Or your preferred model
            tools=gemini_tools
        )
        chat = model.start_chat(enable_automatic_function_calling=False)
        app.logger.info("Gemini client initialized.")

    except ValueError as e:
         app.logger.error(f"Configuration error: {e}")
         # Return a generic error to the client
         return jsonify({"status": "error", "message": f"Server configuration issue: {e}"}), 500
    except Exception as e:
        app.logger.error(f"Error initializing Gemini: {e}")
        return jsonify({"status": "error", "message": "Failed to initialize AI service."}), 500


    # --- Interact with Gemini ---
    try:
        app.logger.info("Sending prompt to Gemini...")
        response = chat.send_message(user_prompt, tools=gemini_tools)
        response_candidate = response.candidates[0]
        response_part = response_candidate.content.parts[0]

        # --- Handle Function Call or Direct Response ---
        if response_part.function_call:
            function_call = response_part.function_call
            function_name = function_call.name
            function_args = dict(function_call.args)
            app.logger.info(f"Gemini requested function: {function_name}")

            if function_name == "update_github_file":
                function_to_call = available_functions[function_name]

                # Prepare arguments, injecting the secure token
                args_dict_with_token = function_args.copy()
                args_dict_with_token["token"] = github_token # Use token loaded from env
                if "branch" not in args_dict_with_token or not args_dict_with_token["branch"]:
                    args_dict_with_token["branch"] = "main"

                # Execute the GitHub update function
                app.logger.info(f"Executing function '{function_name}'...")
                function_response_content = function_to_call(**args_dict_with_token)
                app.logger.info(f"Function result: {function_response_content}")

                # Package result for Gemini
                function_response_part_for_api = glm.Part(
                    function_response=glm.FunctionResponse(
                        name=function_name,
                        response={"content": function_response_content}
                    )
                )

                # Send result back to Gemini
                app.logger.info("Sending function result back to Gemini...")
                second_response = chat.send_message(function_response_part_for_api, tools=gemini_tools)
                final_text = second_response.candidates[0].content.parts[0].text
                app.logger.info(f"Gemini final response: {final_text}")
                # Include the function result status in the final JSON for clarity
                return jsonify({
                    "status": function_response_content.get("status", "unknown"), # success/error from func
                    "gemini_response": final_text,
                    "details": function_response_content.get("message", "")
                }), 200

            else:
                # Handle case where Gemini requests an unknown function
                app.logger.error(f"Gemini requested unknown function: {function_name}")
                return jsonify({
                    "status": "error",
                    "message": f"AI requested an unknown function '{function_name}'."
                }), 501 # Not Implemented

        else:
            # Gemini responded directly without a function call
            direct_response = response_part.text
            app.logger.info(f"Gemini responded directly: {direct_response}")
            return jsonify({
                "status": "success", # Or maybe "info"? Success implies action was taken.
                "gemini_response": direct_response,
                "details": "AI provided a text response without taking action."
            }), 200

    except Exception as e:
        app.logger.exception(f"An error occurred during Gemini interaction: {e}")
        return jsonify({"status": "error", "message": "An unexpected error occurred during AI interaction."}), 500

# ==============================================================================
# === 4. RUN FLASK APP (For local testing) =====================================
# ==============================================================================
if __name__ == "__main__":
    # Use a port number Render might assign or default to 5000/8080
    port = int(os.environ.get('PORT', 8080))
    # host='0.0.0.0' makes it accessible on your network (needed for Render)
    app.run(debug=False, host='0.0.0.0', port=port) # Turn debug=False for production simulation