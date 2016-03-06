/*
 * Copyright 2014 Rodrigo Agerri

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */
package eus.ixa.ixa.pipe.pos.train;


import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.postag.POSContextGenerator;
import opennlp.tools.postag.POSTaggerFactory;
import opennlp.tools.postag.TagDictionary;



/**
 * Extends the POSTagger Factory. Right now we only override the context
 * generators.
 */
public class ClusterFeatureFactory extends POSTaggerFactory {
	protected ClarkCluster ccDictionary;

  /**
   * Creates a {@link ClusterFeatureFactory} that provides the default implementation
   * of the resources.
   */
  public ClusterFeatureFactory() {
  }

  /**
   * Creates a {@link POSTaggerFactory}. Use this constructor to
   * programmatically create a factory.
   * 
   * @param ngramDictionary
   *          the ngrams dictionary
   * @param posDictionary
   *          the postags dictionary
   */
  public ClusterFeatureFactory(final Dictionary ngramDictionary,
      final TagDictionary posDictionary) {
    super(ngramDictionary, posDictionary);
  }
  
  // NEW
  public ClusterFeatureFactory(final Dictionary ngramDictionary,
	      final TagDictionary posDictionary, final ClarkCluster ccDictionary) {
	  this.ngramDictionary = ngramDictionary;
	  this.posDictionary = posDictionary;
	  this.ccDictionary = ccDictionary;/*
	    super(ngramDictionary, posDictionary);*/
	  }
  

  // here new constructor with cluster as parameter --> this in the context generator - first try with one, e.g. 
  /*public ClusterFeatureFactory(final Dictionary ngramDictionary, final TagDictionary posDictionary, final ClarkCluster ccDictionary) {
	  this.ccDictionary = ccDictionary;
  }  */
  /*
   * (non-Javadoc)
   * 
   * @see opennlp.tools.postag.POSTaggerFactory#getPOSContextGenerator()
   */
  @Override
  public final POSContextGenerator getPOSContextGenerator() {
    return new ClusterFeaturesContextGenerator(0, getDictionary());
  }

  /*
   * (non-Javadoc)
   * 
   * @see opennlp.tools.postag.POSTaggerFactory#getPOSContextGenerator(int)
   */
  @Override
  public final POSContextGenerator getPOSContextGenerator(final int cacheSize) {
    return new ClusterFeaturesContextGenerator(cacheSize, getDictionary());
  }

}