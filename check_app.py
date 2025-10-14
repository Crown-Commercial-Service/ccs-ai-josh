from flask import Flask, render_template_string

# Initialize the Flask application
app = Flask(__name__)

# --- HTML Template Content ---
# This string contains the full HTML structure, including the Tailwind CDN
# and the custom CSS needed to replicate the fixed disclaimer bar.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flask Fixed Disclaimer UI</title>
    <!-- Load Tailwind CSS for modern styling and responsiveness -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom CSS to implement the fixed disclaimer bar */
        .fixed-disclaimer {
            position: fixed; /* Locks the bar relative to the viewport */
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 1rem 0; /* Padding for height */
            font-size: 0.9rem;
            z-index: 9999; /* Ensure it stays above all other content */
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1); /* Subtle top shadow */
        }

        /* Ensure main content doesn't hide behind the fixed bar when scrolling */
        .main-content {
            min-height: 150vh; /* Forces scrolling for demonstration */
            padding-bottom: 80px; /* Add space equal to the disclaimer bar height */
        }
    </style>
</head>
<body class="bg-gray-50 font-sans">

    <!-- Main Content Area -->
    <div class="main-content container mx-auto p-4 md:p-8 max-w-4xl">
        <header class="text-center py-8">
            <h1 class="text-5xl font-extrabold text-indigo-800">Flask Application Demo</h1>
            <p class="mt-2 text-xl text-gray-600">Scroll down to see the fixed disclaimer.</p>
        </header>

        <section class="space-y-6 text-lg text-gray-700 bg-white p-6 rounded-xl shadow-lg">
            <h2 class="text-3xl font-semibold text-indigo-600">About This Layout</h2>
            <p>
                We've used the power of **CSS `position: fixed`** to ensure the disclaimer stays glued to the bottom of the screen, even as you navigate through the main content. This is a common pattern for important notices or sticky navigation bars in modern web design.
            </p>
            <p>
                Since the disclaimer has been fixed, we added extra padding to the bottom of this main content section to prevent the last lines of text from being obscured by the fixed bar. This ensures a clean user experience across all screen sizes.
            </p>

            <!-- Placeholder content to force scrolling -->
            <div class="pt-8">
                <div class="h-64 bg-indigo-50 rounded-lg p-4 flex items-center justify-center text-indigo-700 font-semibold shadow-md">
                    More Content Placeholder 1
                </div>
                <div class="h-64 mt-4 bg-green-50 rounded-lg p-4 flex items-center justify-center text-green-700 font-semibold shadow-md">
                    More Content Placeholder 2
                </div>
                <div class="h-64 mt-4 bg-yellow-50 rounded-lg p-4 flex items-center justify-center text-yellow-700 font-semibold shadow-md">
                    End of Scrollable Content
                </div>
            </div>
        </section>

    </div>

    <!-- The fixed disclaimer element (replicated from the Streamlit intent) -->
    <div class="fixed-disclaimer bg-gray-100 border-t border-gray-200 text-gray-600">
         Disclaimer: AI-generated content may not always be accurate or up-to-date. Please verify critical information independently.
    </div>

</body>
</html>
"""
# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main page with the fixed disclaimer UI."""
    # Use render_template_string to serve the HTML content directly
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)
# Note: The application will run when the environment executes this file.
# We omit the if __name__ == '__main__': block for canvas execution environments.
