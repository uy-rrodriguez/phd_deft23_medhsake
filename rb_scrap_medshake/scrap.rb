#!/usr/bin/env ruby

require "json"
require "rubygems"
require "nokogiri"
require "open-uri"


YEARS_IGNORED = [2008]
YEARS_NORD_SUD = 2002.upto(2011).to_a
URL_BASE = "https://www.medshake.net/pharmacie/concours-internat/annales/qcm/voir"

DEBUG = false
if DEBUG
  YEARS = [2015]
  NUMBERS = [39]
  SLEEP = 5
else
  YEARS = 1991.upto(2023)
  NUMBERS = 1.upto(60)
  SLEEP = 0.1
end


def check(year, suffix, question_nbr)
  # Why is 2004-sud Q 21 ignored?
  #
  # if year == 2004
  #   if suffix == "sud"
  #     if question_nbr == 21
  #       return false
  #     end
  #   end
  # end

  # Year 2008 does not exist
  if YEARS_IGNORED.include? year
    return false
  end

  return true
end


def launch

  puts "["

  YEARS.each do |year|
    ["", "nord", "sud"].each do |suffix|
      if suffix == "" and YEARS_NORD_SUD.include? year
        next
      elsif suffix != "" and not YEARS_NORD_SUD.include? year
        next
      end

      NUMBERS.each do |nbr|
        if not check(year, suffix, nbr)
          next
        end

        year_suffix = "#{year}"
        if suffix != ""
          year_suffix = "#{year}-#{suffix}"
          url = "#{URL_BASE}/#{year}-#{suffix}/#{nbr}/"
        else
          url = "#{URL_BASE}/#{year}/#{nbr}/"
        end
        $stderr.puts "#{year_suffix} #{nbr}"

        doc = Nokogiri::HTML(URI.open(url))

        # Extract question cleaning the text, removing "&nbsp;"
        question = doc.css("div.qcm.row").text.strip.split("\n")[3].chomp.gsub(" ", " ").strip
        $stderr.puts question


        # Parse student responses
        rates = {}
        doc.css("table.table.table-sm.tabGraphComp tr").each do |tr|
            tds = tr.css("td")
            if tds.size == 3
              # puts "#{tds[0].children.map(&:text)[-1].strip};#{tds[1].text.strip};#{tds[2].text.strip}"
              answer = tds[0].children.map(&:text)[-1].strip.downcase
              nb_answer = Integer(tds[1].text.strip) rescue 0
              score = Float(tds[2].text.strip) rescue 0.0
              # puts "#{answer};#{nb_answer};#{score}"
              rates[answer] = {
                "nb_answer": nb_answer,
                "score": score,
              }
            end
        end


        # Parse topics
        topics = []
        topics_tags = doc.css("div.container div.row div.col button")
        topics_tags.each do |topic|
          topics.push(topic.text.downcase)
        end


        # Print data as JSON
        data = {
          "question": question,
          "medshake": rates,
          "year": year,
          "year_txt": year_suffix,
          "question_nbr": nbr,
          "topics": topics
        }
        puts JSON.generate(
          data,
          opts={indent: "    ", space: " ", object_nl: "\n", array_nl: "\n"},
        ) + ","


        # puts "------"
        sleep(SLEEP)
      end
    end
  end

  puts "]"

end


def errarg
    puts "Usage : ./programme.rb"
    puts "Mickael Rouvier <mickael.rouvier@gmail.com>"
end


if ARGV.size == 0
    launch
else
    errarg
end
