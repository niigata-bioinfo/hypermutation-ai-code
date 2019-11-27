TILE_DIR = ARGV[0]
DIST_DIR = ARGV[1]
LABEL_DIR = ARGV[2]

system("mkdir -p #{DIST_DIR}/train/tumor")
system("mkdir -p #{DIST_DIR}/train/non-tumor")
system("mkdir -p #{DIST_DIR}/valid/tumor")
system("mkdir -p #{DIST_DIR}/valid/non-tumor")

tumor_coords = {}
Dir.glob("#{LABEL_DIR}/*.txt").each do |path|
  slide_id = path.split('/').last.gsub(/\.txt$/, '')
  tumor_coords[slide_id] = []
  File.open(path).each_line do |l|
    vals = l.chomp.split("\t")
    x = vals[0].to_i
    y = vals[1].to_i
    label = vals[2].to_s
    tumor_coords[slide_id].push("#{x}-#{y}") if label == 'tumor'
  end
end

tumor_count = 0
nt_list = []
Dir.glob("#{TILE_DIR}/*/*.jpg").each do |tile_path|
  destination = rand(9) > 7 ? 'valid' : 'train'
  slide_id = tile_path.split('/')[-2]
  f_name = tile_path.split('/').last
  params = f_name.gsub(/^.*\-/, '').split('.')
  x = params[0].gsub(/^x/, '').to_i
  y = params[1].gsub(/^y/, '').to_i
  if tumor_coords[slide_id].include?("#{x}-#{y}")
    system("cp #{tile_path} #{DIST_DIR}/#{destination}/tumor/#{slide_id}_#{f_name}")
    tumor_count += 1
  else
    nt_list.push(tile_path)
  end
end

nt_list.sample(tumor_count * 4).each do |tile_path|
  destination = rand(9) > 7 ? 'valid' : 'train'
  slide_id = tile_path.split('/')[-2]
  f_name = tile_path.split('/').last
  system("cp #{tile_path} #{DIST_DIR}/#{destination}/non-tumor/#{slide_id}_#{f_name}")
end
