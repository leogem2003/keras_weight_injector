#!/bin/sh

trap 'rm models_pt_temp.zip models_tf_temp.zip' EXIT


https://polimi365-my.sharepoint.com/:u:/g/personal/10566467_polimi_it/EVFcCuMJp9xFoaIjXOm9R48BEE2eNAUQYQxXEB2aYbHP2Q?e=rqodum
https://polimi365-my.sharepoint.com/:u:/g/personal/10566467_polimi_it/EVoOK3heMwVCuI-v5Y0dTvQBUllWuu5Lqd11x4bL3EiM6w?e=NKmgFv

echo 'Downloading PyTorch models'
# models pt
curl 'https://polimi365-my.sharepoint.com/personal/10566467_polimi_it/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2F10566467%5Fpolimi%5Fit%2FDocuments%2FImportantData%2Fmodels%5Fpt%2Ezip' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Referer: https://polimi365-my.sharepoint.com/personal/10566467_polimi_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F10566467%5Fpolimi%5Fit%2FDocuments%2FImportantData%2Fmodels%5Fpt%2Ezip&parent=%2Fpersonal%2F10566467%5Fpolimi%5Fit%2FDocuments%2FImportantData&ga=1' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: iframe' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Connection: keep-alive' -H 'Cookie: rtFa=azv1JIcWsLSi3OS5064+T6TlJVw8mE0MPjigMKtB1AcmOWY0NTJhM2YtM2E2NC00YjYxLTgzNDAtMjExMWRhNGEyMTQ0IzEzMzU3MjE0MjAzNTg3MTUzNSM4Njk3MWRhMS00MGU5LTgwMDAtOWQwZC1iNThmNmNiYzU1NGUjMTA1NjY0NjclNDBwb2xpbWkuaXQjMTg4NTgwI0VQNHVieGJETWZfeVlCOUFtMTBWU2xPQmVTOBgCP0KtDsLamvopr8ck4TRzxULcvr5+0rPoxNVpJka/ofv3nJe3gRh7mDN6qnXgIu4J1+NB8TsSv6tqQel4GymyWKOlFaSk8VWSntgEW1SW+IX6EEkjkhDzMvF4YRUXfkxm49tK7OlRC2hWZyn9tAsHjyAythuhGum6fzS7FwxVhLqNl8enP5viT7x/7cLyXERgxtGZf0XNkWhzJJKxrqOaE1rqaEsMNSOVmEod/9OmvXhKVRCmbAkqae1T+q150VxyikuC6COF0JB529zaRG7/Wbo53XINiPQ42kWp7ilJ9EUtTvewzdOjqaFAt38NdyUW7zOFOHLLW1N5vnRKhC+1AAAA; SIMI=eyJzdCI6MH0=; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEzLDBoLmZ8bWVtYmVyc2hpcHwxMDAzM2ZmZjk2ZDNlOGVjQGxpdmUuY29tLDAjLmZ8bWVtYmVyc2hpcHwxMDU2NjQ2N0Bwb2xpbWkuaXQsMTMzNDUxNDI1NjUwMDAwMDAwLDEzMzM5MzI5NDQyMDAwMDAwMCwxMzM1NzY0NjIwMzU3MTUyNzEsMTMxLjE3NS4yOC4xOTcsMyw5ZjQ1MmEzZi0zYTY0LTRiNjEtODM0MC0yMTExZGE0YTIxNDQsLGQ0MmNlMjEwLWIzZDktNGJkMy1hNTU4LTdlYTgxODhlMDdiMyw4Njk3MWRhMS00MGU5LTgwMDAtOWQwZC1iNThmNmNiYzU1NGUsODY5NzFkYTEtNDBlOS04MDAwLTlkMGQtYjU4ZjZjYmM1NTRlLCwwLDEzMzU3MzAwNjAzNTU1ODk1MiwxMzM1NzQ3MzQwMzU1NTg5NTIsLCxleUo0YlhOZlkyTWlPaUpiWENKRFVERmNJbDBpTENKNGJYTmZjM050SWpvaU1TSXNJbkJ5WldabGNuSmxaRjkxYzJWeWJtRnRaU0k2SWpFd05UWTJORFkzUUhCdmJHbHRhUzVwZENJc0luVjBhU0k2SWxOSFIwbFZSRVV0Wm10SGQzaE1UbGxUVjBKZlFVRWlmUT09LDI2NTA0Njc3NDM5OTk5OTk5OTksMTMzNTcyMTQyMDMwMDAwMDAwLDllNTE3MDJmLWU3ZTktNDE3Yy1hY2IwLTliNGJhNTI3N2ZlMywsLCwsLDAsLDE4ODU4MCxEYURBZmpRUW1weU9YeDJScktfVzVsdm9MWjQsRkVHek1ST2tWVUl4bWlxMTJEMC9JcjExRnpVczhMSlpXd253aXNWUzZETHovVDdyMHlTSFZvK016dmVodDJ6ZUtzN1N5VzQ4SDJuamxjbURIdmlnV1N3Nyt2RnFEWlZFa1NrOUZSNHBpSERLSDhHYlo0L0hzUGovUFlYbTI5NnVuUnVCeGI0OGVmWURCVjRxcUcxMUNJaVBzdW1TZnBnRStndVFJdlc5VktGOWNOZU5MMUU2Ym9KK1pocktzemNYZkk3b3hpemcyM01ZZTZCOS9KWjlZTXBHWVpSZUhaVEtidTEveEFTT1I2dzZmM2YwcjBOTzd1NzZxYmpLL0p1NFJkWE84ZUF5MEM5SzdQNFIwcDl2Y0h2cm55Mmo0RzZVRXNuaU81RGNkVGZBalBwZW5MU2htWGNvVjhIeTNZQ0J1dDVCM0VCbTBDSXduaEo1MEdFVzl3PT08L1NQPg==; odbn=1; MicrosoftApplicationsTelemetryDeviceId=506bf485-561a-4c6a-9e27-2f204d33712e; ai_session=ha/phRkwgSVyBsvl5cs6KL|1712740604304|1712741698027; MSFPC=GUID=c5d3bdec3d2241c4b34d47a4c9849435&HASH=c5d3&LV=202404&V=4&LU=1712740606434' --output models_pt_temp.zip

unzip models_pt_temp.zip -d benchmark_models

echo 'Downloading Tensorflow models'
# models tf
curl 'https://polimi365-my.sharepoint.com/personal/10566467_polimi_it/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2F10566467%5Fpolimi%5Fit%2FDocuments%2FImportantData%2Fmodels%5Fkeras%2Ezip' -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br' -H 'Referer: https://polimi365-my.sharepoint.com/personal/10566467_polimi_it/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F10566467%5Fpolimi%5Fit%2FDocuments%2FImportantData%2Fmodels%5Fkeras%2Ezip&parent=%2Fpersonal%2F10566467%5Fpolimi%5Fit%2FDocuments%2FImportantData&ga=1' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: iframe' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Connection: keep-alive' -H 'Cookie: rtFa=azv1JIcWsLSi3OS5064+T6TlJVw8mE0MPjigMKtB1AcmOWY0NTJhM2YtM2E2NC00YjYxLTgzNDAtMjExMWRhNGEyMTQ0IzEzMzU3MjE0MjAzNTg3MTUzNSM4Njk3MWRhMS00MGU5LTgwMDAtOWQwZC1iNThmNmNiYzU1NGUjMTA1NjY0NjclNDBwb2xpbWkuaXQjMTg4NTgwI0VQNHVieGJETWZfeVlCOUFtMTBWU2xPQmVTOBgCP0KtDsLamvopr8ck4TRzxULcvr5+0rPoxNVpJka/ofv3nJe3gRh7mDN6qnXgIu4J1+NB8TsSv6tqQel4GymyWKOlFaSk8VWSntgEW1SW+IX6EEkjkhDzMvF4YRUXfkxm49tK7OlRC2hWZyn9tAsHjyAythuhGum6fzS7FwxVhLqNl8enP5viT7x/7cLyXERgxtGZf0XNkWhzJJKxrqOaE1rqaEsMNSOVmEod/9OmvXhKVRCmbAkqae1T+q150VxyikuC6COF0JB529zaRG7/Wbo53XINiPQ42kWp7ilJ9EUtTvewzdOjqaFAt38NdyUW7zOFOHLLW1N5vnRKhC+1AAAA; SIMI=eyJzdCI6MH0=; FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEzLDBoLmZ8bWVtYmVyc2hpcHwxMDAzM2ZmZjk2ZDNlOGVjQGxpdmUuY29tLDAjLmZ8bWVtYmVyc2hpcHwxMDU2NjQ2N0Bwb2xpbWkuaXQsMTMzNDUxNDI1NjUwMDAwMDAwLDEzMzM5MzI5NDQyMDAwMDAwMCwxMzM1NzY0NjIwMzU3MTUyNzEsMTMxLjE3NS4yOC4xOTcsMyw5ZjQ1MmEzZi0zYTY0LTRiNjEtODM0MC0yMTExZGE0YTIxNDQsLGQ0MmNlMjEwLWIzZDktNGJkMy1hNTU4LTdlYTgxODhlMDdiMyw4Njk3MWRhMS00MGU5LTgwMDAtOWQwZC1iNThmNmNiYzU1NGUsODY5NzFkYTEtNDBlOS04MDAwLTlkMGQtYjU4ZjZjYmM1NTRlLCwwLDEzMzU3MzAwNjAzNTU1ODk1MiwxMzM1NzQ3MzQwMzU1NTg5NTIsLCxleUo0YlhOZlkyTWlPaUpiWENKRFVERmNJbDBpTENKNGJYTmZjM050SWpvaU1TSXNJbkJ5WldabGNuSmxaRjkxYzJWeWJtRnRaU0k2SWpFd05UWTJORFkzUUhCdmJHbHRhUzVwZENJc0luVjBhU0k2SWxOSFIwbFZSRVV0Wm10SGQzaE1UbGxUVjBKZlFVRWlmUT09LDI2NTA0Njc3NDM5OTk5OTk5OTksMTMzNTcyMTQyMDMwMDAwMDAwLDllNTE3MDJmLWU3ZTktNDE3Yy1hY2IwLTliNGJhNTI3N2ZlMywsLCwsLDAsLDE4ODU4MCxEYURBZmpRUW1weU9YeDJScktfVzVsdm9MWjQsRkVHek1ST2tWVUl4bWlxMTJEMC9JcjExRnpVczhMSlpXd253aXNWUzZETHovVDdyMHlTSFZvK016dmVodDJ6ZUtzN1N5VzQ4SDJuamxjbURIdmlnV1N3Nyt2RnFEWlZFa1NrOUZSNHBpSERLSDhHYlo0L0hzUGovUFlYbTI5NnVuUnVCeGI0OGVmWURCVjRxcUcxMUNJaVBzdW1TZnBnRStndVFJdlc5VktGOWNOZU5MMUU2Ym9KK1pocktzemNYZkk3b3hpemcyM01ZZTZCOS9KWjlZTXBHWVpSZUhaVEtidTEveEFTT1I2dzZmM2YwcjBOTzd1NzZxYmpLL0p1NFJkWE84ZUF5MEM5SzdQNFIwcDl2Y0h2cm55Mmo0RzZVRXNuaU81RGNkVGZBalBwZW5MU2htWGNvVjhIeTNZQ0J1dDVCM0VCbTBDSXduaEo1MEdFVzl3PT08L1NQPg==; odbn=1; MicrosoftApplicationsTelemetryDeviceId=506bf485-561a-4c6a-9e27-2f204d33712e; ai_session=ha/phRkwgSVyBsvl5cs6KL|1712740604304|1712742162492; MSFPC=GUID=c5d3bdec3d2241c4b34d47a4c9849435&HASH=c5d3&LV=202404&V=4&LU=1712740606434' --output models_tf_temp.zip

unzip models_tf_temp.zip -d benchmark_models

echo 'Done'
exit 0