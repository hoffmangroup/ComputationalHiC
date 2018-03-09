# SiameseNN - Computational prediction of chromatin long-range interactions #

SiameseNN: Siamese neural networks for prediction of cell type specific long-range interactions in chromatin from only DNase datasets


## Summary ##
Chromatin is the combination of DNA and proteins that form chromosomes within the nucleus of human cells, and has a significant role in gene expression. In gene expression, transcriptional regulation depends on physical interactions between enhancers and promoters, often not adjacent on the DNA backbone, but rather interacting in the folded 3D conformation of chromatin. Unfortunately, these 3D long-range interactions are not easy to find. Current molecular biology techniques such as chromosome conformation capture (3C) and Hi-C are too expensive for widespread use. To solve this problem, we propose a machine-learning approach based upon a siamese neural network model, previously used for the detection of forged hand-written signatures. This artificial neural network learns the mathematical representation of pairs of chromosome region DNase profiles and states whether they represent a long-range interaction. We tested the effectiveness of our method through a standard deep learning optimization approach, by validating our predicted interactions in a recent Hi-C dataset. Our method is not only able to predict long-range interactions from input dataset made of several cell types, but it is also able to effectively predict interactions for a cell type that is absent from the input dataset.

## Dependencies ##
SiameseNN can be used on any Linux or macOS machine.
To run SiameseNN, you need to have the following programs and packages installed in your machine:

* **Torch** (version 7)
* **PostgreSQL** 

## Database installation ##
Here are the instructions to install the project database on your computer.

Download the database sql file:
`wget https://www.pmgenomics.ca/hoffmanlab/proj/SiameseNN/davide_dnase_hic_database_2018-03-08.sql`

Create the username `davide`:
`sudo su - postgres`
`psql`
` CREATE USER davide WITH PASSWORD 'test' CREATEDB CREATEUSER;`
` \q`
` exit`
 
 Create the database `davide`:
`psql -d postgres -U davide`
` CREATE DATABASE davide;`
` \q`
 
Recover the database from the sql backup file:
`psql davide < davide_dnase_hic_database_2018-03-08.sql`
 
Conditions to use the database through luasql
If the following command works:
psql -d davide -U davide -h localhost
 
Edit the first lines of the `database_management.lua` file this way:
`DB_NAME = "davide"`
`DB_USER = "davide"`
`DB_PWD = "test"`
`DB_ADDRESS = "localhost"`

