#usage generate_learning_set.rb ../slide_list.tsv

SLIDE_LIST_FILE = ARGV[0]

class Donor
  attr_reader :is_hyper
  def initialize(id, is_hyper)
    @id = id
    @tissues = {}
    @is_hyper = is_hyper
  end

  def addSlide(slide, tissue)
    @tissues[tissue] = [] if @tissues[tissue].nil?
    @tissues[tissue].push(slide)
  end

  def getTissueUniqueSlides()
    res = []
    @tissues.each do |tissue, slide|
      res.push(slide.sample(1)[0])
    end

    res
  end
end

donor_list = {}
File.open(SLIDE_LIST_FILE).each do |l|
  next if l[0, 1] == '#'
  vals = l.chomp.split("\t")
  slide_name = vals[0]
  donor_id = vals[1]
  tissue_id = vals[2]
  is_hyper = vals[3] == '1' ? true : false

  donor = donor_list[donor_id].nil? ? Donor.new(donor_id, is_hyper) : donor_list[donor_id]
  donor.addSlide(slide_name, tissue_id)

  donor_list[donor_id] = donor
end

hyper_donor_list = donor_list.keys.select{|k| donor_list[k].is_hyper == true}
non_hyper_donor_list = donor_list.keys.select{|k| donor_list[k].is_hyper == false}

th_id_list = hyper_donor_list.sample(28)
vh_id_list = hyper_donor_list - th_id_list
tn_id_list = non_hyper_donor_list.sample(100)
vn_id_list = (non_hyper_donor_list - tn_id_list).sample(20)

th_id_list.each do |id|
  donor_list[id].getTissueUniqueSlides().each do |slide|
    puts("#{slide}\ttraining_hyper")
  end
end

tn_id_list.each do |id|
  donor_list[id].getTissueUniqueSlides().each do |slide|
    puts("#{slide}\ttraining_non_hyper")
  end
end

vh_id_list.each do |id|
  donor_list[id].getTissueUniqueSlides().each do |slide|
    puts("#{slide}\tvalidation_hyper")
  end
end

vn_id_list.each do |id|
  donor_list[id].getTissueUniqueSlides().each do |slide|
    puts("#{slide}\tvalidation_non_hyper")
  end
end
