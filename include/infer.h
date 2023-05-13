#ifdef __cplusplus
extern "C"
{
#endif

    int inferCifar(const char *model_path, const char *image_path);
    int inferDebias(const char *model_path, const char *rec_path);

#ifdef __cplusplus
}
#endif
