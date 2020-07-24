DeCAPTCHA_CS771  
===============
#### A machine learning &amp; OpenCV based solution to a DeCAPTCHA problem.  

This project was created as a solution to an assignment(3rd) for the *2019 Sem-I* offering of **CS771 Introduction to Machine Learning** by *Prof. Purushottam Kar*. Dataset for the problem was provided by the associated faculty.  
  
**Problem Statement :** CAPTCHAs (Completely Automated Public Turing test to tell Computers and Humans Apart) are popular ways of preventing bots from attempting to logon to systems by extensively searching the password space. In its traditional form, an image is given which contains a few characters (sometimes with some obfuscation thrown in). The challenge is to identify what those characters are and in what order.

**Input Format :** Images of size 600 x 150 pixels are used as input. These images will contain a code composed of either 3 or 4 characters, all of which would be in uppercase. The font of all these characters would be the same, as would be the font size. However, each character may be rotated a bit and each character may be rendered with a different color. The background color of each image can also change. However, all background colors are light in shade. Each image also has some stray lines in the background which are of varying thickness, varying color and of a shade darker than that of the background. These stray lines are intended to make the problem more interesting.  

**Output Format :** For each input test image, return the number of characters in the code and the string representation of the code in the image.  

