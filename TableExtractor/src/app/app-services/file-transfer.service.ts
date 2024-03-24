import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { backendUrl, user } from '../app-config';

@Injectable({
  providedIn: 'root'
})

export class FileTransferService {

  constructor(
    private http: HttpClient,
  ) { }

  fileUpload(selectedFile: FileList): Promise<any> {
    var promise = new Promise<any>((resolve, reject) => {
      const formData = new FormData();
      for (let i = 0; i < selectedFile.length; i++) {
        formData.append('files', selectedFile[i], selectedFile[i].name);
      }
      this.http.post(`${backendUrl}/fileUpload/?user=${user}`, formData)
        .subscribe((rep: any) => {
          resolve(rep)
        })

    })
    return promise
  }

  openAll():Promise<any> {
    var promise = new Promise<any>((resolve, reject) => {
      const headers = { 'content-type': 'application/json'}  
      const body = JSON.stringify({
        'user': user
      });
      this.http.post(`${backendUrl}/openAll`, body, {'headers':headers})
        .subscribe((rep: any) => {
          resolve(rep)
        })

    })
    return promise
  }
}
