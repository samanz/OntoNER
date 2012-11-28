package edu.umass.cs.iesl.ontoner

import scala.io.Source
import cc.factorie._
import cc.factorie.app.nlp._
import cc.factorie.app.nlp.ner._

object ConllChainNer extends edu.umass.cs.iesl.ontoner.ChainNer2 {
  def main(args: Array[String]): Unit = {
  	import cc.factorie.util.DefaultCmdOptions

    object opts extends DefaultCmdOptions {
      val trainFile =     	new CmdOption("train", "eng.train", "FILE", "CoNLL formatted training file.")
      val testFile  =     	new CmdOption("test",  "eng.testb", "FILE", "CoNLL formatted test file.")
      val sigmaSq  =     	new CmdOption("sigmaSq",  "10", "REAL", "Value for regularization constant for BP.")
      val modelDir =      	new CmdOption("model", "chainner.factorie", "DIR", "Directory for saving or loading model.")
      val runXmlDir =     	new CmdOption("run-xml", "xml", "DIR", "Directory for reading NYTimes XML data on which to run saved model.")
      val runPlainFiles = 	new CmdOption("run-plain", List("ner.txt"), "FILE...", "List of files for reading plain texgt data on which to run saved model.")
      val lexiconDir =    	new CmdOption("lexicons", "lexicons", "DIR", "Directory containing lexicon files named cities, companies, companysuffix, countries, days, firstname.high,...")
      val brownClusFile = 	new CmdOption("brown", "", "FILE", "File containing brown clusters.")
      val extended = 		new CmdOption("extended", "Turn on 2 stage feature.")
      val justTest = 		new CmdOption("justTest", "No Training, only Testing.")
	//val noSentences=new CmdOption("nosentences", "Do not use sentence segment boundaries in training.  Improves accuracy when testing on data that does not have sentence boundaries.")
    }
    opts.parse(args)

    val lexes = List("WikiArtWork.lst", "WikiArtWorkRedirects.lst", "WikiCompetitionsBattlesEvents.lst", "WikiCompetitionsBattlesEventsRedirects.lst", "WikiFilms.lst", "WikiFilmsRedirects.lst", "WikiLocations.lst", "WikiLocationsRedirects.lst", "WikiManMadeObjectNames.lst", "WikiManMadeObjectNamesRedirects.lst", "WikiOrganizations.lst", "WikiOrganizationsRedirects.lst", "WikiPeople.lst", "WikiPeopleRedirects.lst", "WikiSongs.lst", "WikiSongsRedirects.lst", "cardinalNumber.txt", "currencyFinal.txt", "known_corporations.lst", "known_country.lst", "known_jobs.lst", "known_name.lst", "known_names.big.lst", "known_nationalities.lst",  "known_state.lst", "known_title.lst", "measurments.txt", "ordinalNumber.txt", "temporal_words.txt")

    aggregate = true
    twoStage = true
    bP = true
    ss = opts.sigmaSq.value.toDouble

    if (opts.lexiconDir.wasInvoked) {
      for(filename <- lexes) {
        println("Reading lexicon "+filename)
        lexicons += new Lexicon(opts.lexiconDir.value+"/"+filename)
        val source = Source.fromFile(opts.lexiconDir.value+"/"+filename)
        for(line <- source.getLines()) {
          if(wordToLex.contains(line.toLowerCase)) {
            wordToLex(line.toLowerCase) = wordToLex(line.toLowerCase) ++ List(filename);
          } else {
            wordToLex(line.toLowerCase) = List(filename);
          }
        }
      }
    }

    if( opts.brownClusFile.wasInvoked) {
          println("Reading brown cluster file " + opts.brownClusFile.value)
          for(line <- Source.fromFile(opts.brownClusFile.value).getLines()){
              val splitLine = line.split("\t")
              clusters(splitLine(1)) = splitLine(0)
          }
    }
    
   if(opts.justTest.wasInvoked) { 
		model.load(opts.modelDir.value)
		test(opts.testFile.value)
	} else {
      train(opts.trainFile.value, opts.testFile.value)
      if (opts.modelDir.wasInvoked) {
		model.save(opts.modelDir.value)
	  }
    }
  }
}