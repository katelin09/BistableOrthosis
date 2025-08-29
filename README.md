# Personalized Bistable Orthoses for Rehabilitation of Finger Joints

Project page: https://interactive-structures.org/publications/2025-09-bistable-orthosis/

## Running the Website Locally

This section guides end users on running the orthosis web interface locally to view and generate brace models using your finger measurements.

### 1. Clone the Repository

Copy the repository URL from the Code button and run in a terminal (VS Code terminal or your system terminal):

```python
git clone https://github.com/yourusername/ModelSimulation.git
cd ModelSimulation/orthosis-website
```

This copies the project to your computer and moves into the website folder.

### 2. Install Dependencies

Check if Node.js and npm are installed:

``` python
node -v
npm -v
```

If both commands print version numbers, you’re ready.

If not, download and install Node.js (npm comes bundled) from nodejs.org.

Then install the project dependencies:
```python
npm install
```

### 3. Start the Server

Run the Node.js server:

```python
node server.js
```

You should see output like:
```python
Server running at http://localhost:3000
```

If you see errors, follow the instructions in the error messages to install missing packages.

### 4. Open the Website

Open a web browser and navigate to the URL shown (e.g., http://localhost:3000
).

Enter your finger dimensions and click Generate.

Wait for the optimized brace model to be computed.

Congrats! Now you can click Download to get the STL mesh file ready for 3D printing.

## Publication

Yuyu Lin, Dian Zhu, Anoushka Naidu, Kenneth Yu, Deon Harper, Eni Halilaj, Douglas Weber, Deborah Ellen Kenney, Adam J. Popchak, Mark Baratz, Alexandra Ion. 2025. Personalized Bistable Orthoses for Rehabilitation of Finger Joints. In Proceedings of UIST ’25. Busan, Republic of Korea. Sept. 28 - Oct. 01, 2025. DOI: https://doi.org/10.1145/3746059.3747643 
