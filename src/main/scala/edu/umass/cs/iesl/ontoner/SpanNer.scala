package edu.umass.cs.iesl.ontoner

import cc.factorie._
import cc.factorie.optimize._
import cc.factorie.app.nlp._
import cc.factorie.app.nlp.ner._
import cc.factorie.util.DefaultCmdOptions
import java.io.File
import scala.io.Source
import cc.factorie.app.chain._
import cc.factorie.app.strings._
import la.{DenseTensor2, DenseTensor1, SparseBinaryTensorLike1}

abstract class SpanNerTemplate extends DotTemplate2[SpanNerLabel,NerSpan] {
  //override def statisticsDomains = ((SpanNerFeaturesDomain, Conll2003NerDomain))
  lazy val weights = new la.DenseTensor2(Conll2003NerDomain.size, SpanNerFeaturesDomain.dimensionSize) // TODO This ordering seems backwards
  def unroll1(label:SpanNerLabel) = Factor(label, label.span)
  def unroll2(span:NerSpan) = Factor(span.label, span)
}

class SpanNerModel extends CombinedModel(
    // Bias term on each individual label 
    new DotTemplateWithStatistics1[SpanNerLabel] {
      lazy val weights = new la.DenseTensor1(Conll2003NerDomain.size)
    },
    // Token-Label within Span
    new SpanNerTemplate { 
      override def statistics(v1:SpanNerLabel#Value, v2:NerSpan#Value) = {
        // TODO Yipes, this is awkward, but more convenient infrastructure could be build.
        //var vector = new cc.factorie.la.SparseBinaryVector(v._1.head.value.length) with CategoricalVectorValue[String] { val domain = SpanNerFeaturesDomain }
        val firstToken = v2.head
        var vector = firstToken.attr[SpanNerFeatures].value.blankCopy
        for (token <- v2; featureIndex <- token.attr[SpanNerFeatures].tensor.activeDomain.asSeq) 
          vector.asInstanceOf[SparseBinaryTensorLike1] += featureIndex // TODO This is shifing array pieces all over; slow.  Fix it.
        v1 outer vector 
      }
    },
    // First Token of Span
    new SpanNerTemplate { 
      override def statistics(v1:SpanNerLabel#Value, v2:NerSpan#Value) = v1 outer v2.head.attr[SpanNerFeatures].value
    },
    // Last Token of Span
    new SpanNerTemplate { 
      override def statistics(v1:SpanNerLabel#Value, v2:NerSpan#Value) = v1 outer v2.last.attr[SpanNerFeatures].value
    },
    // Token before Span
    new SpanNerTemplate { 
      override def unroll1(label:SpanNerLabel) = if (label.span.head.hasPrev) Factor(label, label.span) else Nil
      override def unroll2(span:NerSpan) = if (span.head.hasPrev) Factor(span.label, span) else Nil
      override def statistics(v1:SpanNerLabel#Value, v2:NerSpan#Value) = v1 outer v2.head.prev.attr[SpanNerFeatures].value
    },
    // Token after Span
    new SpanNerTemplate { 
      override def unroll1(label:SpanNerLabel) = if (label.span.last.hasNext) Factor(label, label.span) else Nil
      override def unroll2(span:NerSpan) = if (span.last.hasNext) Factor(span.label, span) else Nil
      override def statistics(v1:SpanNerLabel#Value, v2:NerSpan#Value) = v1 outer v2.last.next.attr[SpanNerFeatures].value
    },
    // Single Token Span
    new SpanNerTemplate { 
      override def unroll1(label:SpanNerLabel) = if (label.span.length == 1) Factor(label, label.span) else Nil
      override def unroll2(span:NerSpan) = if (span.length == 1) Factor(span.label, span) else Nil
      override def statistics(v1:SpanNerLabel#Value, v2:NerSpan#Value) = v1 outer v2.head.attr[SpanNerFeatures].value
    }
    //new SpanLabelTemplate with DotStatistics2[Token,Label] { def statistics(span:Span, label:Label) = if (span.last.hasNext && span.last.next.hasNext) Stat(span.last.next.next, span.label) else Nil },
    // Span Length with Label
    //new SpanLabelTemplate with DotStatistics2[SpanLength,Label] { def statistics(span:Span, label:Label) = Stat(span.spanLength, span.label) },
    // Label of span that preceeds or follows this one
    //new Template2[Span,Span] with Statistics2[Label,Label] { def unroll1(span:Span) = { val result = Nil; var t = span.head; while (t.hasPrev) { if } } }
  ) {
  def this(file:File) = { this(); this.load(file.getPath)}
}

// The training objective
class SpanNerObjective extends TemplateModel(
  new TupleTemplateWithStatistics2[SpanNerLabel, NerSpan] {
    //def statisticsDomains = ((NerSpanDomain, SpanNerLabelDomain))
    def unroll1(label:SpanNerLabel) = Factor(label, label.span)
    def unroll2(span:NerSpan) = Factor(span.label, span)
    def score(labelValue:SpanNerLabel#Value, spanValue:NerSpan#Value) = {
      var result = 0.0
      var trueLabelIncrement = 10.0
      var allTokensCorrect = true
      for (token <- spanValue) {
        //if (token.trueLabelValue != "O") result += 2.0 else result -= 1.0
        if (token.nerLabel.intValue == labelValue.intValue) {
          result += trueLabelIncrement
          trueLabelIncrement += 2.0 // proportionally more benefit for longer sequences to help the longer seq steal tokens from the shorter one.
        } else if (token.nerLabel.target.categoryValue == "O") {
          result -= 1.0
          allTokensCorrect = false
        } else {
          result += 1.0
          allTokensCorrect = false
        }
        if (token.spans.length > 1) result -= 100.0 // penalize overlapping spans
      }
      if (allTokensCorrect) {
        if (!spanValue.head.hasPrev || spanValue.head.prev.nerLabel.intValue != labelValue.intValue) result += 5.0 // reward for getting starting boundary correct
        if (!spanValue.last.hasNext || spanValue.last.next.nerLabel.intValue != labelValue.intValue) result += 5.0 // reward for getting starting boundary correct
      }
      result
    }
  }
)


class SpanNer {
  var verbose = false
  
  class Lexicon(filename:String) extends cc.factorie.app.chain.Lexicon(filename) {
    def name = filename.substring(filename.lastIndexOf('/')+1).toUpperCase
  }
  val lexicons = new scala.collection.mutable.ArrayBuffer[Lexicon]
  val model = new edu.umass.cs.iesl.ontoner.SpanNerModel
  val objective = new edu.umass.cs.iesl.ontoner.SpanNerObjective
  val predictor = new SpanNerPredictor(model)
  val clusters = new scala.collection.mutable.HashMap[String,String]
  val wordToLex = new scala.collection.mutable.HashMap[String,List[String]]
  var aggregate = false
  var count = 0
  var didagg = false

  def initFeatures(document:Document, vf:Token=>CategoricalTensorVar[String]): Unit = {
    //println("Count" + count)
    count=count+1
    import cc.factorie.app.strings.simplifyDigits
    for (token <- document.tokens) {
      didagg = false
	  val features = vf(token)
      val rawWord = token.string
      val word = simplifyDigits(rawWord).toLowerCase
      features += "W="+word
      if (token.isCapitalized) features += "CAPITALIZED"
      else features += "NOTCAPITALIZED"
      if (token.isPunctuation) features += "PUNCTUATION"
      for(lexicon <- wordInLex(token)) features += "LEX="+lexicon
      if (clusters.size > 0 && clusters.contains(rawWord)) {
        features += "CLUS="+prefix(4,clusters(rawWord))
        features += "CLUS="+prefix(6,clusters(rawWord))
        features += "CLUS="+prefix(10,clusters(rawWord))
        features += "CLUS="+prefix(20,clusters(rawWord))
      }
    }
    for (sentence <- document.sentences)
      cc.factorie.app.chain.Observations.addNeighboringFeatureConjunctions(sentence.tokens, vf, "^[^@]*$", List(0), List(1), List(2), List(-1), List(-2))
      document.tokens.foreach(t => if (t.string.matches("[A-Za-z]+")) vf(t) ++= t.charNGrams(2,5).map(n => "NGRAM="+n))
      document.tokens.foreach(t => vf(t) ++= t.prevWindow(4).map(t2 => "PREVWINDOW="+simplifyDigits(t2.string).toLowerCase))
      document.tokens.foreach(t => vf(t) ++= t.nextWindow(4).map(t2 => "NEXTWINDOW="+simplifyDigits(t2.string).toLowerCase))

    for(token <- document.tokens) {
      aggregateContext(token, vf)
    }
  }

  def prevWindowNum(t:Token, n:Int): IndexedSeq[(Int,Token)] = t.prevWindow(n).map(x => (t.prevWindow(n).indexOf(x),x)).toIndexedSeq
  def nextWindowNum(t:Token, n:Int): IndexedSeq[(Int,Token)] = t.nextWindow(n).map(x => (t.nextWindow(n).indexOf(x),x)).toIndexedSeq

  def prefix( prefixSize : Int, cluster : String ) : String = {
    if(cluster.size > prefixSize)
         cluster.substring(0, prefixSize)
    else
         cluster
  }

  def subsect(phrase : Array[String], maxOutLength : Int) : List[String] = {
    val middle = (phrase.size/2)
    var keys = List[String]()
    for(i <- (0 to maxOutLength)) {
      var start = middle
      for(j <- (0 to i)) {
        start = middle-j;
        var key : String= ""
        if(start > -1 && (start+i) < phrase.size) {
          for(k <- (0 to i)) {
            key =  key + " " + phrase(start+k)
          }
          keys = key.trim().toLowerCase :: keys
        }
      }
    }
    keys
  }

  def locate(token : Token, key : String) : String = {
	val string = token.string
	val split = key.split(" ")
	if(split.length == 1 && key.trim.toLowerCase == string.trim.toLowerCase) return "U-"
	val last = split.length-1
	for(i <- 0 until split.length) {
		if(split(i).toLowerCase == string.toLowerCase) i match {
			case 0 => return "B-"
			case last => return "L-"
		}
	}
        "I-"
  }

  def wordInLex(token : Token) : List[String] = {
    if(wordToLex.size > 0) {
      var checkWords = List()
      val fullLength = 15
      val fullPhrase = new Array[String](fullLength)
      fullPhrase(fullLength/2) = token.string
      var count = 0

      var prevToken = token;
      while(count < (fullLength/2) && prevToken.hasPrev) {
        count = count+1
        prevToken = prevToken.prev
        fullPhrase(fullLength/2-count) = prevToken.string
      }

      count = 0
      var nextToken = token;
      while(count < (fullLength/2) && nextToken.hasNext) {
        count = count+1
        nextToken = nextToken.next
        fullPhrase(fullLength/2+count) = nextToken.string
      }

      val keys = subsect(fullPhrase, 7)

      var lexes = List[String]()
      for(key <- keys) {
        if(wordToLex.contains(key))
          lexes = wordToLex(key).map(locate(token, key) + _) ::: lexes
      }
      return lexes
    } else
      return List()
  }
  
  def addContextFeatures[A<:Observation[A]](t : Token, from : Token, vf:Token=>CategoricalTensorVar[String]) : Unit = {
    didagg = true
    vf(t) ++= prevWindowNum(from,2).map(t2 => "CONTEXT="+simplifyDigits(t2._2.string).toLowerCase + "@-" + t2._1)
    vf(t) ++= nextWindowNum(from, 2).map(t2 => "CONTEXT="+simplifyDigits(t2._2.string).toLowerCase + "@" + t2._1)
  

    for(t2 <- prevWindowNum(from,2)) {
	if(clusters.contains(t2._2.string)) { 
			vf(t) += ("CONTEXTPATH="+prefix(4, clusters(t2._2.string)) + ("@-" + t2._1.toString))
    	}
    }

    for(t2 <- nextWindowNum(from, 2)) {
	if(clusters.contains(t2._2.string)) {
			vf(t) += ("CONTEXTPATH="+prefix(4, clusters(t2._2.string)) + ("@" + t2._1.toString))
		}
    }
  }
  
  def aggregateContext[A<:Observation[A]](token : Token, vf:Token=>CategoricalTensorVar[String]) : Unit = {
    var count = 0
    var compareToken : Token = token
    while(count < 200 && compareToken.hasPrev) {
      count += 1
      compareToken = compareToken.prev
      if(token.string.toLowerCase == compareToken.string.toLowerCase)
        addContextFeatures(token, compareToken, vf)
    }
    count = 0
    compareToken = token
    while(count < 200 && compareToken.hasNext) {
      count += 1
      compareToken = compareToken.next
      if(token.string.toLowerCase == compareToken.string.toLowerCase)
        addContextFeatures(token, compareToken, vf)
    }
  }


  def train(trainFiles:Seq[String], testFile:String): Unit = {
    predictor.verbose = true
    // Read training and testing data.  The function 'featureExtractor' function is defined below.  Now training on seq == whole doc, not seq == sentece
    val trainDocuments = trainFiles.flatMap(LoadConll2003.fromFilename(_))
    val testDocuments = LoadConll2003.fromFilename(testFile)
    println("Read "+trainDocuments.flatMap(_.sentences).size+" training sentences, and "+testDocuments.flatMap(_.sentences).size+" testing ")

  	(trainDocuments ++ testDocuments).foreach(_.tokens.map(token => token.attr += new SpanNerFeatures(token)))

    trainDocuments.foreach(initFeatures(_,(t:Token)=>t.attr[SpanNerFeatures]))
    testDocuments.foreach(initFeatures(_,(t:Token)=>t.attr[SpanNerFeatures]))

    println("Have "+trainDocuments.map(_.length).sum+" trainTokens "+testDocuments.map(_.length).sum+" testTokens")
    println("FeaturesDomain size="+SpanNerFeaturesDomain.dimensionSize)
    println("LabelDomain "+Conll2003NerDomain.toList)
    
    //if (verbose) trainDocuments.take(10).map(_.tokens).flatten.take(500).foreach(token => { print(token.string+"\t"); printFeatures(token) })
    
    // The learner
    val sampler = new TokenSpanSampler(model, objective) {
      //logLevel = 1
      temperature = 0.01
      override def preProcessHook(t:Token): Token = { 
        super.preProcessHook(t)
        if (t.isCapitalized) { // Skip tokens that are not capitalized
          if (verbose) t.spansOfClass[NerSpan].foreach(s => println({if (s.isCorrect) "CORRECT " else "INCORRECT "}+s))
          // Skip this token if it has the same spans as the previous token, avoiding duplicate sampling
          //if (t.hasPrev && t.prev.spans.sameElements(t.spans)) null.asInstanceOf[Token] else
          t 
        } else null.asInstanceOf[Token] 
      }
      override def proposalsHook(proposals:Seq[Proposal]): Unit = {
        if (verbose) { proposals.foreach(p => println(p+"  "+(if (p.modelScore > 0.0) "MM" else "")+(if (p.objectiveScore > 0.0) "OO" else ""))); println }
        super.proposalsHook(proposals)
      }
    }
    val learner = new SampleRankTrainer(sampler, new MIRA)
    
    
    // Train!
    for (i <- 1 to 11) {
      println("Iteration "+i) 
      // Every third iteration remove all the predictions
      if (i % 3 == 0) { println("Removing all spans"); (trainDocuments ++ testDocuments).foreach(_.clearSpans(null)) }
      learner.processContexts(trainDocuments.map(_.tokens).flatten)
      //learner.learningRate *= 0.9
      predictor.processAll(testDocuments.map(_.tokens).flatten)
      println("*** TRAIN OUTPUT *** Iteration "+i); if (verbose) { trainDocuments.foreach(printDocument _); println; println }
      println("*** TEST OUTPUT *** Iteration "+i); if (verbose) { testDocuments.foreach(printDocument _); println; println }
      println ("Iteration %2d TRAIN EVAL ".format(i)+evalString(trainDocuments))
      println ("Iteration %2d TEST  EVAL ".format(i)+evalString(testDocuments))
    }

  }  
  def evalString(documents:Seq[Document]): String = {
    var trueCount = 0
    var predictedCount = 0
    var correctCount = 0
    for (document <- documents) {
      predictedCount += document.spans.length
      document.spansOfClass[NerSpan].foreach(span => if (span.isCorrect) correctCount += 1)
      for (token <- document) {
        val tokenTargetCategory = token.nerLabel.target.categoryValue
        if (tokenTargetCategory != "O" && (!token.hasPrev || token.prev.nerLabel.target.categoryValue != tokenTargetCategory))
           trueCount += 1
      }
    }
    def precision = if (predictedCount == 0) 1.0 else correctCount.toDouble / predictedCount
    def recall = if (trueCount == 0) 1.0 else correctCount.toDouble / trueCount
    def f1 = if (recall+precision == 0.0) 0.0 else (2.0 * recall * precision) / (recall + precision)
    "OVERALL f1=%-6f p=%-6f r=%-6f".format(f1, precision, recall)
  }
  
  def printDocument(document:Document): Unit = {
    for (token <- document) {
      token.startsSpansOfClass[NerSpan].foreach(span => print("<"+span.label.value+">"))
      print(token.string)
      token.endsSpansOfClass[NerSpan].foreach(span => print("</"+span.label.value+">"))
      print(" ")
    }
    println
    for (span <- document.spansOfClass[NerSpan].sortForward(span => span.start.toDouble)) {
      println("%s len=%-2d %-8s %-15s %-30s %-15s".format(
          if (span.isCorrect) " " else "*",
          span.length,
          span.label.value, 
          if (span.hasPredecessor(1)) span.predecessor(1).string else "<START>", 
          span.phrase, 
          if (span.hasSuccessor(1)) span.successor(1).string else "<END>"))
    }
  }


  def printToken(token:Token) : Unit = {
    //print("printToken "+token.word+"  ")
    val spans = token.spansOfClass[NerSpan]
    for (span <- spans) {
      println("%s %-8s %-15s %-30s %-15s".format(
          if (span.isCorrect) " " else "*",
          span.label.value, 
          if (span.hasPredecessor(1)) span.predecessor(1).string else "<START>", 
          span.phrase, 
          if (span.hasSuccessor(1)) span.successor(1).string else "<END>"))
      span.foreach(token => print(token.string+" ")); println
    }
  }
}

object SpanNer extends SpanNer {
  // The "main", examine the command line and do some work
  def main(args: Array[String]): Unit = {
    // Parse command-line
    object opts extends DefaultCmdOptions {
      val trainFile = new CmdOption("train", List("eng.train"), "FILE", "CoNLL formatted training file.")
      val testFile  = new CmdOption("test",  "", "FILE", "CoNLL formatted dev file.")
      val modelDir =  new CmdOption("model", "spanner.factorie", "DIR", "Directory for saving or loading model.")
      val runXmlDir = new CmdOption("run", "xml", "DIR", "Directory for reading NYTimes XML data on which to run saved model.")
      val lexiconDir =new CmdOption("lexicons", "lexicons", "DIR", "Directory containing lexicon files named cities, companies, companysuffix, countries, days, firstname.high,...") 
      val noSentences=new CmdOption("nosentences", "Do not use sentence segment boundaries in training.  Improves accuracy when testing on data that does not have sentence boundaries.")
      val brownClusFile = new CmdOption("brown", "", "FILE", "File containing brown clusters.")
      val aggregateTokens = new CmdOption("aggregate", "Turn on context aggregation feature.")
      val extended = new CmdOption("extended", "Turn on 2 stage feature.")
    }
    opts.parse(args)
    
    val lexes = List("WikiArtWork.lst", "WikiArtWorkRedirects.lst", "WikiCompetitionsBattlesEvents.lst", "WikiCompetitionsBattlesEventsRedirects.lst", "WikiFilms.lst", "WikiFilmsRedirects.lst", "WikiLocations.lst", "WikiLocationsRedirects.lst", "WikiManMadeObjectNames.lst", "WikiManMadeObjectNamesRedirects.lst", "WikiOrganizations.lst", "WikiOrganizationsRedirects.lst", "WikiPeople.lst", "WikiPeopleRedirects.lst", "WikiSongs.lst", "WikiSongsRedirects.lst", "cardinalNumber.txt", "currencyFinal.txt", "known_corporations.lst", "known_country.lst", "known_jobs.lst", "known_name.lst", "known_names.big.lst", "known_nationalities.lst",  "known_state.lst", "known_title.lst", "measurments.txt", "ordinalNumber.txt", "temporal_words.txt")

    if (opts.lexiconDir.wasInvoked) {
      //for (filename <- List("cities", "companies", "companysuffix", "countries", "days", "firstname.high", "firstname.highest", "firstname.med", "jobtitle", "lastname.high", "lastname.highest", "lastname.med", "months", "states")) {
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
    
    train(opts.trainFile.value, opts.testFile.value)
    if (opts.modelDir.wasInvoked) model.save(opts.modelDir.value)   
  }
}

