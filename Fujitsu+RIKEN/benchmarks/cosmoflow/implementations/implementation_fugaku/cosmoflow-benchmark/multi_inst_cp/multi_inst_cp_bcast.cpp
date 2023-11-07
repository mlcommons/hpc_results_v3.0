#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/time.h>
#include <future>
#include <sys/resource.h>
#include <openssl/md5.h>

#define MAX_STR_LEN 1024
#define MAX_FILE_SIZE (192*1024*1024)

int world_rank, world_size;
const char kinds[2][16] = {"train", "validation"};

#define COMP_NONE 0
#define COMP_XZ   1


#define USE_ASYNC

void abort_mpi(const char err[]);

double get_time_ms(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (double)tv.tv_usec/1000000;
}

#define LOG(...)				\
  do{ \
    printf("[%d] %.3f ", world_rank, get_time_ms());	\
    printf(__VA_ARGS__); \
    printf("\n"); \
    fflush(stdout); \
  }while(0);

void exec(const char command[]) {
  LOG("%s", command);
  int r = system(command);
  if (r != 0) {
    MPI_Abort(MPI_COMM_WORLD, r);
  }
}

void get_file_name(const char data_dir[], const char kind[], int index, int comp_type, char * out_filename)
{
  int r;
  switch (comp_type) {
  case COMP_NONE:
    {
      r = sprintf(out_filename, "%s/%s/%d/cosmo_%s_%d.tar", data_dir, kind, index/1000, kind, index);
      break;
    }
  case COMP_XZ:
    {
      char ext[] = "xz";
      r = sprintf(out_filename, "%s/%s/%d/cosmo_%s_%d.tar.%s", data_dir, kind, index/1000, kind, index, ext);
      break;
    }
  default:
    abort_mpi("get_file_name: invalid comp_type");
  }
  
  if (r == EOF) {
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
}

void make_tmp_file_name(const char dir[], const char kind[], int index, int comp_type, const char last[], char * out_filename)
{
  int r;
  switch (comp_type) {
  case COMP_NONE:
    {
      r = sprintf(out_filename, "%s/cosmo_%s_%d.tar%s", dir, kind, index, last);
      break;
    }
  case COMP_XZ:
    {
      char ext[] = "xz";
      r = sprintf(out_filename, "%s/cosmo_%s_%d.tar.%s%s", dir, kind, index, ext, last);
      break;
    }
  default:
    abort_mpi("make_tmp_file_name: invalid comp_type");
  }
}

void exec_cp(const char src[], const char dst[])
{
  char command[MAX_STR_LEN];

  sprintf(command, "cp %s %s", src, dst);
  exec(command);
}

void exec_tar(const char src_dir[], const char dst_dir[], const int comp_type, const int begin, const int end)
{
  char filename[MAX_STR_LEN];
  char command[MAX_STR_LEN];

  const char no_decomp[] = "";
  const char xz_decomp[] = "-I pixz";
  const char *comp_option;

  switch (comp_type) {
  case COMP_NONE:
    {
      comp_option = no_decomp;
      break;
    }
  case COMP_XZ:
    {
      comp_option = xz_decomp;
      break;
    }
  default:
    abort_mpi("exec_tar: invalid comp_type");
  }

  for (int k = 0; k < 2; k++) {
    const char *kind = kinds[k];

    for (int i = begin; i < end; i += 1) {
      get_file_name(src_dir, kind, i, comp_type, filename);

      sprintf(command, "tar %s -xf %s -C %s/%s", comp_option, filename, dst_dir, kind);

      exec(command);
    }
  }
}

void exec_llio(const char src_dir[], const int purge, const int comp_type, const int begin, const int end)
{
  char filename[MAX_STR_LEN];
  char command[MAX_STR_LEN];

  for (int k = 0; k < 2; k++) {
    const char *kind = kinds[k];

    for (int i = begin; i < end; i += 1) {
      get_file_name(src_dir, kind, i, comp_type, filename);

      sprintf(command, "llio_transfer %s %s", (purge ? "--purge" : ""), filename);

      exec(command);
    }
  }
}

int get_local_begin(const int rank, const int size, const int begin , const int end)
{
  int num = end - begin;
  int num_local = num / size;
  return begin + num_local * rank;
}

int get_local_end(const int rank, const int size, const int begin , const int end)
{
  int num = end - begin;
  int num_local = num / size;
  return begin + num_local * (rank + 1);
}

int exist_file(const char filename[]){
  struct stat buffer;
  int r = stat(filename, &buffer);
  return (r == 0);
}

void touch_file(const char filename[])
{
  FILE *fp = fopen(filename, "w");
  fclose(fp);
}

size_t get_file_size(const char filename[])
{
  struct stat sb;

  if (stat(filename, &sb) == -1) {
    printf("error stat\n");
    return 0;
  }

  return sb.st_size;
}

void abort_mpi(const char err[])
{
  printf("[%d] abort:%s\n", world_rank, err);
  fflush(stdout);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

void print_md5_sum(unsigned char* md) {
  int i;
  for(i=0; i <MD5_DIGEST_LENGTH; i++) {
    printf("%02x",md[i]);
  }
}

size_t read_file(const char filename[], char *buffer)
{
  size_t filesize = get_file_size(filename);
  //printf("%s: %zd bytes\n", src_filename, filesize);
  if (filesize > MAX_FILE_SIZE) {
    return 0;
  }

  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    return 0;
  }
  LOG("fread (%s)", filename);
  if (fread(buffer, 1, filesize, fp) != filesize) {
    return 0;
  }
  fclose(fp);

  //unsigned char result[MD5_DIGEST_LENGTH];
  //MD5((unsigned char*) buffer, filesize, result);
  //LOG("fread_end (%02x%02x, %zd, %s)", result[0], result[1], filesize, filename);
  LOG("fread_end (%zd, %s)", filesize, filename);

  return filesize;
}		 

int get_comp_type(const char comp_type_str[])
{
  if (strcmp(comp_type_str, "none") == 0) {
    return COMP_NONE;
  }
  if (strcmp(comp_type_str, "xz") == 0) {
    return COMP_XZ;
  }

  abort_mpi("get_comp_type: invalid comp_type");
  return -1;
}


int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

  const int num_arguments = 8;

  //argument num_inst
  if (argc != (num_arguments + 1)) {
    LOG("invalid # of arguments");
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  int argv_idx = 1;
  const char *src_dir = argv[argv_idx++];
  const char *dst_dir = argv[argv_idx++];
  const char *tmp_dir = argv[argv_idx++];

  const char *comp_type_str = argv[argv_idx++];
  const int total_num_tars = atoi(argv[argv_idx++]);
  const int num_inst = atoi(argv[argv_idx++]);
  const char *begin_flag_filename = argv[argv_idx++];
  const char *end_flag_filename = argv[argv_idx++];

  if (argv_idx != (num_arguments + 1) ){
    abort_mpi("some arguments were not processed");
  }

  const int comp_type = get_comp_type(comp_type_str);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const int inst_size = world_size / num_inst;  //属するインスタンスのサイズ
  const int inst_rank = world_rank % inst_size; //属するインスタンス内のランク
  const int inst_num = world_rank / inst_size;  //属するインスタンスの番号
  //printf("inst_size=%d, inst_rank=%d, inst_num=%d\n", inst_size, inst_rank, inst_num);
  
  // wait begin flag
  LOG("waiting flag...");
  fflush(stdout);
  while(! exist_file(begin_flag_filename)) {
    sleep(1);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);

  for (int k = 0; k < 2; k++) {
    char command[MAX_STR_LEN];
    sprintf(command, "mkdir -p %s/%s", dst_dir, kinds[k]);
    exec(command);
  }


  //MPIのコミュニケータを作る
  MPI_Comm bcast_comm;
  MPI_Comm_split(MPI_COMM_WORLD, inst_rank, inst_num, &bcast_comm);

  {
    //test
    int bcast_comm_size, bcast_comm_rank;
    MPI_Comm_size(bcast_comm, &bcast_comm_size);
    MPI_Comm_rank(bcast_comm, &bcast_comm_rank);
    if (bcast_comm_size != num_inst || bcast_comm_rank != inst_num) {
      abort_mpi("invalid bcast_comm");
    }
  }

  char (*buffers)[MAX_FILE_SIZE] = (char (*)[MAX_FILE_SIZE])malloc(sizeof(char)*MAX_FILE_SIZE*2);
  if (buffers == NULL) {
    abort_mpi("malloc failed");
  }


  char src_filenames[2][MAX_STR_LEN];
#ifdef USE_ASYNC
  std::future<size_t> res[2];
#else
  size_t res[2];
#endif
  
  int num_tars = total_num_tars / inst_size; //割り切れないといけない
  for (int k = 0; k < 2; k++) {
    const int tar_idx_begin = num_tars * inst_rank;
    const int tar_idx_end = num_tars * (inst_rank+1);
    for (int i = tar_idx_begin; i < tar_idx_end; i++) {
      const int bcast_root = 0;

      int buf_idx = i%2;
      char *src_filename = src_filenames[buf_idx];
      
      if (i == tar_idx_begin) {
	get_file_name(src_dir, kinds[k], i, comp_type, src_filename);

	if (inst_num == bcast_root) {
	  #ifdef USE_ASYNC
	  res[buf_idx] = std::async(std::launch::async, read_file, src_filename, buffers[buf_idx]);
	  #else
	  res[buf_idx] = read_file(src_filename, buffers[buf_idx]);
	  #endif
	}
      }

      unsigned long long filesize = 0;
      if (inst_num == bcast_root) {
	LOG("Wait (%s)", src_filename);
	#ifdef USE_ASYNC
	filesize = res[buf_idx].get();
	#else
	filesize = res[buf_idx];
	#endif
	if (filesize == 0) {
	  abort_mpi("fread error");
	}
      }

      if (i + 1 < tar_idx_end) {
	int next_buf_idx = (i+1)%2;
	char *src_filename = src_filenames[next_buf_idx];
	get_file_name(src_dir, kinds[k], i+1, comp_type, src_filename);

	if (inst_num == bcast_root) {
	  #ifdef USE_ASYNC
	  res[next_buf_idx] = std::async(std::launch::async, read_file, src_filename, buffers[next_buf_idx]);
	  #else
	  res[next_buf_idx] = read_file(src_filename, buffers[next_buf_idx]);
	  #endif
	}
      }
      
      //filesizeのbcast
      LOG("Bcast size(%llu) (%s)", filesize, src_filename);
      MPI_Bcast(&filesize, 1, MPI_UNSIGNED_LONG_LONG, bcast_root, bcast_comm);
      
      //dataのbcast
      LOG("Bcast data (%s)", src_filename);
      char *buffer = buffers[buf_idx];
      MPI_Bcast(buffer, filesize, MPI_BYTE, bcast_root, bcast_comm);


      if(0){ //pipeを使う版はうまくいかない
	char command[MAX_STR_LEN];
	sprintf(command, "tar -x -C %s/%s", dst_dir, kinds[k]);
	//sprintf(command, "tee %s/%s/%d", dst_dir, kinds[k], i);

	FILE *fp = popen(command, "w");
	if (fp == NULL) {
	  abort_mpi("popen failed");
	}
	LOG("fwrite (%s)", "pipe");
	size_t offset = 0;
	while(offset < filesize) {
	  size_t wrotesize = fwrite(buffer + offset, 1, filesize - offset, fp);
	  LOG("fwrite (writesize: %zd)", wrotesize);
	  fflush(stdout);
	  if (wrotesize == 0) {
	    abort_mpi("fwrite error");
	  }
	  offset += wrotesize;
	}
	int r = pclose(fp);
	if (r != 0) {
	  LOG("pclose (r : %d)", r);
	  fflush(stdout);
	  abort_mpi("pclose error");
	}
      }

      else{
      char tmp_filename[MAX_STR_LEN];
      make_tmp_file_name(tmp_dir, kinds[k], i, comp_type, "", tmp_filename);
      char tmp_filename_wip[MAX_STR_LEN];
      make_tmp_file_name(tmp_dir, kinds[k], i, comp_type, "__", tmp_filename_wip);

      FILE *fp;
      fp = fopen(tmp_filename_wip, "wb");
      if (fp == NULL) {
	abort_mpi("fopen(wb) failed");
      }
      LOG("fwrite (%s)", tmp_filename_wip);
      if (fwrite(buffer, 1, filesize, fp) < filesize) {
	abort_mpi("fwrite error");
      }
      fclose(fp);

      LOG("rename (%s, %s)", tmp_filename_wip, tmp_filename);
      if (rename(tmp_filename_wip, tmp_filename) != 0) {
	abort_mpi("rename failed");
      }

      if (0) {
	//(一応bcastしたファイルのmd5をチェック)
	char command[MAX_STR_LEN];
	sprintf(command, "md5sum %s", tmp_filename);
	exec(command);
      }

      //tarで展開
      if(0){
	char command[MAX_STR_LEN];
	sprintf(command, "tar -xf %s -C %s/%s", tmp_filename, dst_dir, kinds[k]);
	exec(command);
      }
      //一時ファイルを削除
      if(0){
	char command[MAX_STR_LEN];
	sprintf(command, "rm %s", tmp_filename);
	exec(command);
      }
      if(0){
	struct rusage r;
	if (getrusage(RUSAGE_SELF, &r) != 0) {
	  /*Failure*/
	}
	LOG("maxrss=%ld\n", r.ru_maxrss);
	fflush(stdout);
      }
      }
    }
  }

  free(buffers);

  // touch end flag
  // touch_file(end_flag_filename);

  MPI_Finalize();

  LOG("done!");

  return 0;
}
