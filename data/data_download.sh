mkdir opinions
mkdir people
mkdir clusters

wget "https://www.courtlistener.com/api/bulk-data/opinions/scotus.tar.gz"
tar xf scotus.tar.gz -C ./opinions
rm scotus.tar.gz

wget "https://www.courtlistener.com/api/bulk-data/people/all.tar.gz"
tar xf all.tar.gz -C ./people
rm all.tar.gz

wget "https://www.courtlistener.com/api/bulk-data/clusters/scotus.tar.gz"
tar xf scotus.tar.gz -C ./clusters
rm scotus.tar.gz

wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz -P ../model/
gunzip -c ../model/GoogleNews-vectors-negative300.bin.gz > ../model/GoogleNews-vectors-negative300.bin