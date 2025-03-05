# Life Science Technologies (LST)-project

LST project - Domain adaptation of microscopic cell images

Contributors: Anni Niskanen, Juhani Kolehmainen, Lauri Pohjola, Antti Huttunen, and Tuomas Poutanen

Check the project report from ```src/report/```

## Dataset

The Dataset called LIVECell is utilised for training the model. LIVECell is high-quality, manually annotated and expert-validated dataset. It contains over 1.6 million cells from different cell morphologies. Additional information about LIVECell can be found [here](https://sartorius-research.github.io/LIVECell/).

<details><summary><b> How to download LIVECell dataset? </b> </summary><br />

Dataset is stored in an Amazon Web Services (AWS) S3-bucket. If you have **an AWS IAM-user using the AWS-CLI**, you can download dataset using terminal command:

`aws s3 sync s3://livecell-dataset .`

**Otherwise follow these steps:**

Use `curl` to make an HTTP-request to get the S3 XML-response and save to `files.xml`:

```
curl -H "GET /?list-type=2 HTTP/1.1" \
     -H "Host: livecell-dataset.s3.eu-central-1.amazonaws.com" \
     -H "Date: 20161025T124500Z" \
     -H "Content-Type: text/plain" http://livecell-dataset.s3.eu-central-1.amazonaws.com/ > files.xml

```

After that, get the urls from files in `ulrs.txt` using `grep`:

```grep -oPm1 "(?<=<Key>)[^<]+" files.xml | sed -e 's/^/http:\/\/livecell-dataset.s3.eu-central-1.amazonaws.com\//' > urls.txt```

Finally, download the wanted files using `wget`.

Full instructions for downloading the LIVECell and file structures of dataset can be found [here](https://sartorius-research.github.io/LIVECell/#:~:text=Download%20all%20of%20LIVECell). 
</details>

## Connect aalto linux server for training the neural net

In linux terminal use `ssh` command to connect server. Working servers are `lyta.aalto.fi` and `kosh.aalto.fi`. You can contact servers by runing the for example command:

`ssh <Aalto_username>@lyta.aalto.fi`

After you have connected server, you need to connect computer having GPU. Those computers locate in CS-building's Paniikki classroom.

The names of computers in Paniikki:

befunge, bit, bogo, brainfuck, deadfish, emo, entropy, false, fractran, fugue, glass, haifu, headache, intercal, malbolge, numberwang, ook, piet, regexpl, remorse, rename, shakespeare, smith, smurf, spaghetti, thue, unlambda, wake, whenever, whitespace, zombie

You can connect computer by runing `ssh` command in server. Example:

`ssh befunge`
