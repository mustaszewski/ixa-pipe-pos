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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.postag.POSContextGenerator;
import opennlp.tools.util.Cache;
import opennlp.tools.util.StringList;

//import eus.ixa.ixa.pipe.nerc.dict.ClarkCluster;
import java.io.FileWriter; // DEBUG ONLY
import java.io.IOException; //DEBUG ONLY


/**
 * An improved context generator for the POS Tagger. This baseline generator
 * provides more contextual features such as bigrams to the
 * {@code @DefaultPOSContextGenerator}. These extra features require at least
 * 2GB memory to train, more if training data is large.
 * 
 * @author ragerri
 * @version 2014-07-08
 */
public class ClusterFeaturesContextGenerator implements POSContextGenerator {

  /**
   * The ending string.
   */
  private final String SE = "*SE*";
  /**
   * The starting string.
   */
  private final String SB = "*SB*";
  /**
   * Default prefix length.
   */
  private static final int PREFIX_LENGTH = 4;
  /**
   * Default suffix length.
   */
  private static final int SUFFIX_LENGTH = 4;
  /**
   * Has capital regexp.
   */
  private static Pattern hasCap = Pattern.compile("[A-Z]");
  /**
   * Has number regexp.
   */
  private static Pattern hasNum = Pattern.compile("[0-9]");
  /**
   * The context Cache.
   */
  private Cache contextsCache;
  /**
   * The words key.
   */
  private Object wordsKey;
  /**
   * The tag dictionary.
   */
  private final Dictionary dict;
  /**
   * The dictionary ngrams.
   */
  private final String[] dictGram;
  
  static ClarkCluster clarkCluster;
  
  /**
   * The Clark Class of the Token
   */
  //private Map<String, String> clarkAttributes;

  /**
   * Initializes the current instance.
   * 
   * @param aDict
   *          the dictionary
   */
  public ClusterFeaturesContextGenerator(final Dictionary aDict) {
    this(0, aDict);
  }

  /**
   * Initializes the current instance.
   * 
   * @param cacheSize
   *          the cache size
   * @param aDict
   *          the dictionary
   */
  
  public ClusterFeaturesContextGenerator(final int cacheSize, final Dictionary aDict) {
    this.dict = aDict;
    this.dictGram = new String[1];
    //this.clarkCluster = null; // NEW
    if (cacheSize > 0) {
      this.contextsCache = new Cache(cacheSize);
    }
  }
  /*
  public ClusterFeaturesContextGenerator(final int cacheSize, final Dictionary aDict, final ClarkCluster ccDictionary) {
	    this.dict = aDict;
	    this.dictGram = new String[1];
	    this.clarkCluster = ccDictionary;
	    System.out.println("\nclarkCluster in ContextGenerator:\t"+this.clarkCluster+"\n");
	    if (cacheSize > 0) {
	      this.contextsCache = new Cache(cacheSize);
	    }
	  }
  */

  /**
   * Obtain prefixes for each token.
   * 
   * @param lex
   *          the current word
   * @return the prefixes
   */
  protected static String[] getPrefixes(final String lex) {
    final String[] prefs = new String[PREFIX_LENGTH];
    for (int li = 0, ll = PREFIX_LENGTH; li < ll; li++) {
      prefs[li] = lex.substring(0, Math.min(li + 1, lex.length()));
    }
    return prefs;
  }

  /**
   * Obtain suffixes for each token.
   * 
   * @param lex
   *          the word
   * @return the suffixes
   */
  protected static String[] getSuffixes(final String lex) {
    final String[] suffs = new String[SUFFIX_LENGTH];
    for (int li = 0, ll = SUFFIX_LENGTH; li < ll; li++) {
      suffs[li] = lex.substring(Math.max(lex.length() - li - 1, 0));
    }
    return suffs;
  }
  
  /**
   * Obtain Clark Cluster Number for each token
   * 
   */
  protected static String getWordClass(String token) {
	  String clarkClass = clarkCluster.lookupToken(token);
	  if (clarkClass == null) {
		  clarkClass = "noclarkclass";
	  }
	  return clarkClass;
  }
  
  /*
  public String getWordClass(String token) {
	    String clarkClass = clarkCluster.lookupToken(token);
	    if (clarkClass == null) {
	      clarkClass = unknownClarkClass;
	    }
	    return clarkClass;
	  }*/

  /*
   * (non-Javadoc)
   * 
   * @see opennlp.tools.postag.POSContextGenerator#getContext(int,
   * java.lang.String[], java.lang.String[], java.lang.Object[])
   */
  public final String[] getContext(final int index, final String[] sequence,
      final String[] priorDecisions, final Object[] additionalContext) {
    return getContext(index, sequence, priorDecisions);
  }

  /**
   * Returns the context for making a pos tag decision at the specified token
   * index given the specified tokens and previous tags.
   * 
   * @param index
   *          The index of the token for which the context is provided.
   * @param tokens
   *          The tokens in the sentence.
   * @param tags
   *          The tags assigned to the previous words in the sentence.
   * @return The context for making a pos tag decision at the specified token
   *         index given the specified tokens and previous tags.
   */
  public final String[] getContext(final int index, final Object[] tokens,
      final String[] tags) {
    String next, nextnext, lex, prev, prevprev;
    String tagprev, tagprevprev;
    tagprev = tagprevprev = null;
    next = nextnext = lex = prev = prevprev = null;

    lex = tokens[index].toString();
    if (tokens.length > index + 1) {
      next = tokens[index + 1].toString();
      if (tokens.length > index + 2) {
        nextnext = tokens[index + 2].toString();
      } else {
        nextnext = this.SE; // Sentence End
      }

    } else {
      next = this.SE; // Sentence End
    }

    if (index - 1 >= 0) {
      prev = tokens[index - 1].toString();
      tagprev = tags[index - 1]; //removed for CRF Test

      if (index - 2 >= 0) {
        prevprev = tokens[index - 2].toString();
        tagprevprev = tags[index - 2];
      } else {
        prevprev = this.SB; // Sentence Beginning
      }
    } else {
      prev = this.SB; // Sentence Beginning
    }
    final String cacheKey = index + tagprev + tagprevprev;
    if (this.contextsCache != null) {
      if (this.wordsKey == tokens) {
        final String[] cachedContexts = (String[]) this.contextsCache
            .get(cacheKey);
        if (cachedContexts != null) {
          return cachedContexts;
        }
      } else {
        this.contextsCache.clear();
        this.wordsKey = tokens;
      }
    }
    final List<String> featureList = new ArrayList<String>();
    featureList.add("default");
    // add the word itself
    featureList.add("w=" + lex);
    this.dictGram[0] = lex;
    if (this.dict == null || !this.dict.contains(new StringList(this.dictGram))) {
      // do some basic suffix analysis
      final String[] suffs = getSuffixes(lex);
      for (final String suff : suffs) {
        featureList.add("suf=" + suff);
      }

      final String[] prefs = getPrefixes(lex);
      for (final String pref : prefs) {
        featureList.add("pre=" + pref);
      }
      
      // NEW: Get Clark Cluster Class for current Token
      String clarkClass = getWordClass(lex.toLowerCase());
      //String clarkClass = "zero";
      //featureList.add(clarkAttributes.get("dict") + "=" + clarkClass);
      featureList.add("clark=" + clarkClass);
      
      
      // see if the word has any special characters
      if (lex.indexOf('-') != -1) {
        featureList.add("h");
      }

      if (hasCap.matcher(lex).find()) {
        featureList.add("c");
      }

      if (hasNum.matcher(lex).find()) {
        featureList.add("d");
      }
    }
    // add the words and pos's of the surrounding context
    if (prev != null) {
      featureList.add("pw=" + prev);
      // bigram w-1,w
      featureList.add("pw,w=" + prev + "," + lex);
      if (tagprev != null) {
        featureList.add("pt=" + tagprev);
        // bigram tag-1, w
        featureList.add("pt,w=" + tagprev + "," + lex);
      }
      if (prevprev != null) {
        featureList.add("ppw=" + prevprev);
        if (tagprevprev != null) {
          // bigram tag-2,tag-1
          featureList.add("pt2,pt1=" + tagprevprev + "," + tagprev);
        }
      }
    }

    if (next != null) {
      featureList.add("nw=" + next);
      if (nextnext != null) {
        featureList.add("nnw=" + nextnext);

      }
    }
    final String[] contexts = featureList
        .toArray(new String[featureList.size()]);
    if (this.contextsCache != null) {
      this.contextsCache.put(cacheKey, contexts);
    }
    
    // START DEBUG ONLY
	try {
		FileWriter writer = new FileWriter("DebugClusterFeatureContext.txt", true);
		writer.write("Feature List\t"+featureList.toString()+"\n");
		if (tokens != null) {
			writer.write("Tokens:\t");
			for (Object tok : tokens) {
				writer.write(tok.toString()+"  ");
			}
			
		}
		writer.write("\nTags:\t");
		if (tags != null) {
			
			for (String tag : tags) {
				writer.write(tag.toString()+"  ");
			}
			writer.write("\n## "+tags.length +" ##");
		}
		else {
			writer.write("null!");
		}
		
		writer.write("\n\n");   // write new line
		writer.close();
	} catch (IOException e) {
		e.printStackTrace();
    	}
    
    // END DEBUG ONLY
    
    return contexts;
  }

}
