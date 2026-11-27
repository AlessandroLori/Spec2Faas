package main

// Provider S3 support per workloads Go. This file is supplied by Spec2FaaS,
// never generated/edited by the LLM. Real S3 requests go to the isolated
// Floci endpoint using the official AWS SDK for Go v2.

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

func providerS3Client() (*s3.Client, error) {
	endpoint := os.Getenv("AWS_ENDPOINT_URL_S3")
	// Never fall back to a real AWS service during a validation run.
	if endpoint != "http://floci-aws:4566" {
		return nil, fmt.Errorf("Provider S3: missing or unexpected isolated S3 endpoint")
	}
	accessKey, secretKey := os.Getenv("AWS_ACCESS_KEY_ID"), os.Getenv("AWS_SECRET_ACCESS_KEY")
	if accessKey != "test" || secretKey != "test" {
		return nil, fmt.Errorf("Provider S3: unexpected S3 test credentials")
	}
	cfg := aws.Config{
		Region: "us-east-1",
		Credentials: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
			return aws.Credentials{AccessKeyID: accessKey, SecretAccessKey: secretKey, Source: "floci-s3"}, nil
		}),
	}
	return s3.NewFromConfig(cfg, func(options *s3.Options) {
		options.BaseEndpoint = aws.String(endpoint)
		options.UsePathStyle = true
	}), nil
}

// providerS3PutBytes writes the provided bytes to exactly the requested S3 key.
func providerS3PutBytes(bucket, key string, data []byte) error {
	client, err := providerS3Client()
	if err != nil {
		return err
	}
	_, err = client.PutObject(context.Background(), &s3.PutObjectInput{
		Bucket: aws.String(bucket), Key: aws.String(key),
		Body: bytes.NewReader(data), ContentLength: aws.Int64(int64(len(data))),
	})
	return err
}

func providerS3PutFile(bucket, key, filePath string) error {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return err
	}
	return providerS3PutBytes(bucket, key, data)
}

func providerS3GetBytes(bucket, key string) ([]byte, error) {
	client, err := providerS3Client()
	if err != nil {
		return nil, err
	}
	result, err := client.GetObject(context.Background(), &s3.GetObjectInput{
		Bucket: aws.String(bucket), Key: aws.String(key),
	})
	if err != nil {
		return nil, err
	}
	defer result.Body.Close()
	const limit = 128 * 1024 * 1024
	data, err := io.ReadAll(io.LimitReader(result.Body, limit+1))
	if err != nil {
		return nil, err
	}
	if len(data) > limit {
		return nil, fmt.Errorf("Provider S3: S3 object exceeds size limit")
	}
	return data, nil
}

func providerS3GetFile(bucket, key, filePath string) error {
	data, err := providerS3GetBytes(bucket, key)
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, data, 0600)
}

// SeBS-style naming is one possible application-level contract, not an S3
// requirement: insert an independent random eight-hex-digit suffix before ext.
func providerS3UniqueKey(key string) (string, error) {
	var salt [4]byte
	if _, err := rand.Read(salt[:]); err != nil {
		return "", err
	}
	ext := path.Ext(key)
	return strings.TrimSuffix(key, ext) + "." + hex.EncodeToString(salt[:]) + ext, nil
}

func providerS3UploadFile(bucket, key, filePath string) (string, error) {
	unique, err := providerS3UniqueKey(key)
	if err != nil {
		return "", err
	}
	if err := providerS3PutFile(bucket, unique, filePath); err != nil {
		return "", err
	}
	return unique, nil
}

// providerS3ListKeys returns actual object keys, not local paths, including all pages.
func providerS3ListKeys(bucket, prefix string) ([]string, error) {
	client, err := providerS3Client()
	if err != nil {
		return nil, err
	}
	pages := s3.NewListObjectsV2Paginator(client, &s3.ListObjectsV2Input{
		Bucket: aws.String(bucket), Prefix: aws.String(prefix),
	})
	keys := make([]string, 0)
	for pages.HasMorePages() {
		page, err := pages.NextPage(context.Background())
		if err != nil {
			return nil, err
		}
		for _, object := range page.Contents {
			keys = append(keys, aws.ToString(object.Key))
			if len(keys) > 10000 {
				return nil, fmt.Errorf("Provider S3: listing exceeds key limit")
			}
		}
	}
	return keys, nil
}
